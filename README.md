# Model-Based-Tree-Search
Combining an explanatory tree search, which succeeds in planning or learns a better model


When should we recreate our planning? When we encounter a transition that gets to far from our anticipations. Then we either follow our plans successfully, or learn from the model. (it should be possible to get interesting theoretical results)

RL should be used for attention.

Evolving the known reward functions to make the agent optimistic about the untried spaces. (we tell it the state has a positive reward when it hasn’t tried it)
Or creating a curious agent by rewarding the discovery of divergences between the model and the MDP.
