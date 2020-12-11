# CLAD: Curriculum Learning through Distilled Discriminators

This repo is mainly based on https://github.com/ben-eysenbach/sac, rewritten entirely in PyTorch, using PyTorch-Lightning, Hydra and Weights and Biases. Contains the code for the paper *Curriculum Learning through Distilled Discriminators*. A 3-minute explainer can be found [here](https://www.youtube.com/watch?v=1klGyHZ5-2w)

## Abstract

Engineering reward functions to successfully train an agent and generalize to downstream tasks is challenging. Automated Curriculum Learning (ACL) proposes to generate tasks instead. This avoids spurious optima and reduces engineering effort. Recent works in ACL generate policies and tasks jointly such that the policies can be used as an initialization to quickly learn downstream tasks. However, these methods expose the agent to task distribution shifts, which may hurt its generalization capability. We automatically design a curriculum that helps overcome this drawback. First, we show that retraining the agent from scratch on a stationary task distribution improves generalization. Second, given a generated task, we construct an easier task by distilling the reward function into a narrower network reducing reward sparsity. Using this distilled task to form a simple curriculum, we obtain better generalization to downstream tasks compared to our baseline retrained agent.

The full paper can be found [here](./CLAD-camera-ready.pdf)




