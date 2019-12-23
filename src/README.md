# Source Code Structure
The folder `neurips2019` contains a python package, which holds the code for agents, environments, preprocessing and evaluation.

## The `neurips2019` package 
The package contains the following modules:
- agents
  This module holds implementations of different agent algorithms, and some supporting code.
  The file `agent.py` implements an abstract class acting as an interface for all agents to enable more streamlined testing and evaluation.
- environments
  This module holds wrappers for different envrionments. For one the Neurosmash environment but also examples from the OpenAI gym.
  The file `environment.py` implements an abstract class acting as an interface for all environments to enable more streamlined testing and evaluation.
- preprocessing
  This module contains functions that are related to preprocessing. E.g. compression or image alteration algorithms
  There are algorithms to correctly crop and rotate the neurosmash canvas to contain only the stage area, as well as functions to normalize images or build difference images.
  An implementation of an autoencoder is in testing.
- util
  This module contains supporting code that is not necessarily part of any other submodule.

## The rest of the source folder
The loose files in the `src` folder themselves are scripts to run, train or evaulate an agent in an environment and are to be more streamlined in the future.
