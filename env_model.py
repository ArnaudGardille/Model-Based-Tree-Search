from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from scipy import integrate
from scipy.interpolate import interpn,interp1d,RegularGridInterpolator, Akima1DInterpolator
from scipy.optimize import minimize
import scipy.signal as sig
from scipy.optimize import least_squares, minimize

import time
from tqdm import trange
import importlib

#from sklearn.metrics import mean_absolute_error

from numpy import linalg as la

import copy

import sys
from importlib import reload,import_module
import os

import random
import gym

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from sklearn import preprocessing
from torch.utils.data import Dataset

device = 'cpu'
#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class StaticDataset(Dataset):
    def __init__(self, X, Y, scalers :tuple = None):
        self.X = X
        self.Y = Y
        #print(X, Y)
        if scalers is None:
            print('Creating scalers')
            self.scaler_X = preprocessing.StandardScaler().fit(X)
            self.scaler_Y = preprocessing.StandardScaler().fit(Y)
            print('Done')
        else:
            self.scaler_X = scalers[0]
            self.scaler_Y = scalers[1]

        self.scaled_X = self.scaler_X.transform(X)
        self.scaled_Y = self.scaler_Y.transform(Y)
        self.scaled_X = torch.tensor(self.scaled_X).float()
        self.scaled_Y = torch.tensor(self.scaled_Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.scaled_X[idx], self.scaled_Y[idx]
        #torch.tensor(self.scaled_X[idx]).float() , torch.tensor(self.scaled_Y[idx]).float()

    def get_scalers(self):
        return self.scaler_X, self.scaler_Y


    

class DynamicDataset(torch.utils.data.IterableDataset):
    def __init__(self, env :gym.Env, render=True):
        super(DynamicDataset).__init__()
        self.env =env
        self.render = render

    def create_static_datasets(self, sizes :list =[1000, 100]):
        datasets = []
        for i, nb in enumerate(sizes):
            states_actions, diff_states = self.get_samples(nb)
            if i == 0:
                dataset = StaticDataset(states_actions, diff_states)
                scalers = dataset.get_scalers()
            else:
                dataset = StaticDataset(states_actions, diff_states, scalers)
            datasets.append(dataset)

        return datasets

    def get_samples(self, nb):
        states_actions = []
        diff_states = []
        iterable_self = iter(self)
        for i in trange(nb):
            S, A, Y = next(iterable_self)
            X = torch.cat((S, A), dim=0)
            states_actions.append(X[None, :])
            diff_states.append(Y[None, :])

        states_actions = torch.cat(states_actions, dim=0)
        diff_states = torch.cat(diff_states, dim=0)
        return states_actions, diff_states
         
    def __iter__(self):
        done=True
        while True:
            if done:
                observation = self.env.reset()

            if self.render:
                self.env.render()
            action = self.env.action_space.sample()
            #action = np.clip(action, -1.0, 1.0)
            #print(observation)
            next_observation, reward, done, info = self.env.step(action)
            #print(torch.tensor([observation]).shape)
            observation_tensor = torch.tensor(observation).float()
            action_tensor = torch.tensor(action).float()
            diff_observation_tensor = torch.tensor(next_observation).float() - observation_tensor
            #print(observation_tensor, action_tensor, diff_observation_tensor)

            yield observation_tensor, action_tensor, diff_observation_tensor

            observation = next_observation

class Perceptron(nn.Module):

    def __init__(self, state_size, action_size, num_hidden_layers, size__hidden_layer = 200, bias=True):
        super().__init__()
        layer = []
        input_size = state_size + action_size
        for i in range(num_hidden_layers):
            layer.append(nn.Linear(input_size, size__hidden_layer, bias=bias))
            layer.append(nn.ReLU())
            input_size = size__hidden_layer
        layer.append(nn.Linear(size__hidden_layer, state_size, bias=bias))
        self.net = nn.Sequential(*layer)

    def forward(self, X):     
        return self.net(X)

    '''def forward(self, S, A):  
        X = torch.cat((S, A), dim=1)   
        return self.net(X)'''

class EnvModel():
    def __init__(self, state_size, action_size, num_hidden_layers=3) -> None:
        self.model = Perceptron(state_size, action_size, num_hidden_layers=3)

    def predict_diff_states(self, state, action):
        state = state[None,:]
        action = action[None,:]

        # Concatenate the input and output of each block on the channel dimension
        unscaled_X = torch.cat((state, action), dim=1)
        #print('unscaled_X: ', unscaled_X)
        scaled_X = self.scaler_X.transform(unscaled_X)
        scaled_X = torch.tensor(scaled_X).float()
        #print('scaled_X: ', scaled_X)
        scaled_diff = self.model(scaled_X).detach().numpy()
        #print('scaled_diff: ', scaled_diff)
        unscaled_diff = self.scaler_Y.inverse_transform(scaled_diff)
        #print('unscaled_diff: ', unscaled_diff)
        return torch.tensor(unscaled_diff[0]).float()

    def predict_next_state(self, state, action):
        return self.predict_diff_states(state, action) + state

    def save(self):
        pass

    def load(self):
        pass

    def fit(self, list_datasets, num_epochs=25, lr :float =0.01):
        datasets = {'train': list_datasets[0], 'val': list_datasets[1]}

        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=64,
                                                shuffle=True, num_workers=8)
                    for x in ['train', 'val']}
        dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

        self.scaler_X, self.scaler_Y = datasets['train'].get_scalers()
        criterion = nn.MSELoss()

        # Observe that all parameters are being optimized
        optimizer = torch.optim.Adam(self.model.parameters(), lr)

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        writer = SummaryWriter('tensorboard/model_learning')
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10

        pbar = trange(num_epochs)
        for epoch in pbar:
            #print(f'Epoch {epoch}/{num_epochs - 1}')
            #print('-' * 10)

            # Each epoch has a training and validation phase
            phase_loss = {}
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    #print('inputs: ', inputs)
                    #print('labels: ', labels)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        
                        loss = criterion(outputs, labels)

                        #print('outputs: ', outputs)
                        #print('labels: ', labels)
                        #print('loss: ', loss)
                        #print()

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                phase_loss[phase] = epoch_loss
                #print(f'{phase} Loss: {epoch_loss:.4f}')
                #print()

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    
                writer.add_scalar(f'{phase} Loss', epoch_loss, epoch)
                writer.flush()
            
            pbar.set_description("Train Loss: %2.4f ; Val Loss: %2.4f" % (phase_loss['train'], phase_loss['val']))

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Loss: {best_loss:4f}')

        self.model.load_state_dict(best_model_wts)
        print()

    def fit_single_dataset(self, dataset :StaticDataset, num_epochs :int =25, lr :float =0.01):
        self.scaler_X, self.scaler_Y = dataset.get_scalers()
        criterion = nn.MSELoss()

        # Observe that all parameters are being optimized
        optimizer = torch.optim.Adam(self.model.parameters(), lr)

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        writer = SummaryWriter('tensorboard/model_learning')
        since = time.time()

        len_dataset = len(dataset)

        pbar = trange(num_epochs)
        for epoch in pbar:
            running_loss = 0.0
            for state_action, label in dataset:
                #state, action, label = next(dataset)
                #print(state, action, label)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    #print(state_action)
                    output = self.model(state_action)
                    #print(state, action, label, output)
                    loss = criterion(output, label)

                    #print('outputs: ', outputs)
                    #print('labels: ', labels)
                    #print('loss: ', loss)
                    #print()

                    loss.backward()
                    optimizer.step()

                    # statistics
                running_loss += loss.item() 
                

            scheduler.step()

            epoch_loss = running_loss  / len_dataset

            pbar.set_description("Loss: %2.4f" % epoch_loss)


            """print(f'Loss: {epoch_loss:.4f}')
            print()"""

            writer.add_scalar(f' Loss', epoch_loss, epoch)
            writer.flush()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    def visualize(self, dynamic_dataset :DynamicDataset, steps=1000):
        iterable_dataset = iter(dynamic_dataset)

        #columns = ['Position', 'Velocity', 'Angle','Angular Velocity']

        pbar = trange(steps)
        labels = []
        preds = []
        inputs = []
        done=True
        for _ in pbar:
            state, action, label = next(iterable_dataset)
            
            #observation_tensor = torch.tensor(observation).float()
            #action_tensor = torch.tensor([action]).float()[None,:]
            #print(observation_tensor, action_tensor)
            #state_action = torch.cat((state, action), dim=0)[None, :]   
            #pred = self.model(state_action).detach() #.numpy()
            pred = self.predict_diff_states(state, action)
            #print(state, action, label, pred)
            preds.append(pred.numpy())
            labels.append(label.numpy())
            inputs.append(action.numpy())
            
        
        inputs = np.array(inputs)
        labels = np.array(labels)

        preds = np.array(preds)

        #print('inputs: ', inputs)
        #print('labels: ', labels)
        #print('preds: ', preds)
        for i in range(preds.shape[-1]): #, 'velocity']):
            plt.figure(figsize=(12, 12))
            plt.plot(labels[:,i], label = 'label')
            plt.plot(preds[:,i], label = 'prediction')
            plt.legend()
            plt.title('Evolution of state component nb ' + str(i))
            plt.show()

        env.close()


if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0")
    dynamic_dataset = DynamicDataset(env)
    datasets = dynamic_dataset.create_static_datasets([1000, 1000])
    env_model = EnvModel(2, 1)
    #print(dataset[0])
    env_model.fit(datasets)
    env_model.visualize(dynamic_dataset, 100)
    dynamic_dataset.env.close()