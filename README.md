# deepgroebner-pytorch-cocoa

This package contains the following components:

- An environment for S-pair selection in Buchberger's algorithm. This environment is designed according to the design guidelines provided by the [gymnasium](https://gymnasium.farama.org/) library. The algorithm is coded in C++ with the CoCoALib library, resulting in fast rollouts and support for a wide range of inputs to the algorithm. A Python wrapper is provided using Cython so the environment can be used within Python scripts. The environment uses a priority queue data structure to give observations a fixed dimension. How this priority queue structure works is explained in [training-runs.md](docs/tutorials/training-runs.md).
- The package includes some methods for generating inputs to the algorithm for training and the ability for the user to supply a list of training ideals. See [input] 
- A PPO reinforcement learning library for training policies for S-pair selection in Buchberger's algorithm. There are a range of policy architectures to use already included with the package. More details about these architectures are given in [training-runs.md](docs/tutorials/training-runs.md). 
- Scripts for evaluating the performance of trained policies on a fixed test corpus of ideals.
- Scripts for performing hyperparameter sweeps.
- Scripts for evaluating heuristics on fixed test corpora for comparison with trained policies. 

If you have any issues using the library or any features you would like to see added, please submit an issue on this GitHub repo and I'll endeavour to answer at earliest convinence.

Acknowledgements of previous work that helped in the creation of this package in [NOTICE.md](NOTICE.md)

## Installation

Installation instructions are in INSTALL.md

## Tutorials

- [Input Ideals](docs/tutorials/input-ideals.md)
- [Training Runs](docs/tutorials/training-runs.md)
- [Evaluation and Test Corpora](docs/tutorials/eval-corpora-and-greedy-eval.md)
- [Hyperparameter Sweeps](docs/tutorials/hyperparameter-sweeps.md)

## Typical Workflow

1. Choose an ideal distribution to train an S-pair selection policy with (see [Input Ideals](docs/tutorials/input-ideals.md) for details) 
2. Train a policy using proximal policy optimization (PPO) on rollouts of Buchberger's algorithm solving inputs from the input distribution ([Training Runs](docs/tutorials/training-runs.md)) 
3. Generate a separate test corpus of ideals from the training distribution, using a different seed from the evaluation corpus used during training. Evaluate the trained policy by running Buchberger's algorithm on the test corpus using the policy for S-pair selection. ([Evaluation and Test Corpora](docs/tutorials/eval-corpora-and-greedy-eval.md))

## Tracking with W&B

If you want to be able to use your Weights & Biases account to track training runs, make sure to install the optional tracking dependency and log in to Weights and Biases.

