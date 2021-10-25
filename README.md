# Usage

## Environment setup

```
conda env create --name envname --file=environment.yml
```
## Train models in notebook

```
jupyter notebook
```
# Algorithm
## Reward
There are two main modifications for the reward. The first modification alters original env reward by given +10 for discovering new cells and zero otherwise. This way, we can reinforce exploring and as we do not to think about collisions in this problem, we can use zero as default reward. The second modification is based on Intrinsic Curiosity Module from the paper Curiosity-driven Exploration by Self-supervised Prediction. This method uses another network to predict novelty of the state as an error between predicted and actual latent spaces of randomly initialized network. It serves as an additional incentive to explore states even if the environmental reward is 0.

## Learning method

PPO is used as the state-of-the-art on-policy approach.

## Architecture

Three convolutional layers followed up by 2 fully connected layers.

## Framework

Rllib

# Results

## Observations and Improvements

We can see in ```out.gif``` that agent can reliably explore non-trivial rooms and does not wander in already visited states due to curiosity module. Additional improvement can be a LSTM cell to alleviate partial observability in the environment.
## Behaviour

Sample trajectory can be found in ```out.gif```

## Saved model

```checkpoint/*```
