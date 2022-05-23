# Netarkivet-word2vec

Code for training static word and document embeddings on the Danish web archive, Netarkivet.

## Setup:

- Make sure to add a wandb.key file in the current directory containing the key to Weights and Biases
- If you intend to log the results to a different wandb project, make sure to modify the main script (This might change in the future, it will probably either read a file or use command line arguments)
- If the script fails to connect to the wandb project specified in the source code, it will log to stdout, so no worries
- make sure to run **setup.sh** before you run any of the scripts, it will install dependencies and log into wandb

## Training:

To train a model you may run `main.py`.
The script is a full-fledged CLI for training a Word2Vec or a Doc2Vec model on Netarkivet.
To access the documentation run:
<br>
`python3 main.py -h`
<br>
If you want to run the process in the background (e.g. on Ucloud), I recommend using Nohup:
<br>
`nohup python3 -u main.py [args] &`
<br>

## Hyperparameter optimization:

The project structure is properly set up to run sweeps on Weights and Biases.
Beware that it only runs with Word2Vec for now.

WARNING: _optuna_optimization.py is being phased out in favor of Sweeps, and is no longer maintained_

### 1. Create Sweep

`wandb sweep sweep_word2vec.yaml`

### 2. Start an Agent

You run a sweep on multiple machines with the help of agents.
Insert the sweep ID returned by sweep creation.
`wandb agent SWEEP_ID`

## Topic Analysis

To analyse the topics in the corpus you may open `topic_analysis.ipynb`
in Jupyter.
It contains code for:

1. obtaining a random sample from the corpus
2. training either an NMF or an LDA topic model
3. visually inspecting the topics with pyLDAvis
