# Netarkivet-word2vec

Code for training static word and document embeddings on the Danish web archive, Netarkivet.

## Usage

- Make sure to add a wandb.key file in the current directory containing the key to Weights and Biases
- If you intend to log the results to a different wandb project, make sure to modify the main script (This might change in the future, it will probably either read a file or use command line arguments)
- If the script fails to connect to the wandb project specified in the source code, it will log to stdout, so no worries
- make sure to run **setup.sh** before you run any of the scripts, it will install dependencies and log into wandb

## Scripts

### 1. Main:

Main script to train word or paragraph embedding models.
The script is a full blown CLI, you can specify hyperparameters
and other options by using command line arguments and flags.

    positional arguments:
    model                 Specifies which model should be
                          trained on the corpus.
                          (options: {'word2vec', 'doc2vec'}
                          optional, default=word2vec)

    optional arguments:
    -h, --help            show this help message and exit
    -d [DATA_PATH], --data_path [DATA_PATH]
                            Path to the root directory of the data files (optional,
                            default=/work/netarkivet-cleaned/)
    -s SAVE_PATH, --save_path SAVE_PATH
                            Path, where the model is going to be saved and where the
                            model is initialised from
    -p [PREPROCESSING_WORKERS], --preprocessing_workers [PREPROCESSING_WORKERS]
                            Number of processes assigned to preprocess strings for
                            training (optional,default=6)
    -t [TRAINING_WORKERS], --training_workers [TRAINING_WORKERS]
                            Number of processes assigned to train the model
                            (optional,default=6)
    -w [WINDOW_SIZE], --window_size [WINDOW_SIZE]
                            Window size of the work2vec model (optional,default=5)
    -v [VECTOR_SIZE], --vector_size [VECTOR_SIZE]
                            Dimensionality of the desired word vectors
                            (optional,default=100)
    -n, --no_porn_filtering
                            Flag to turn of porn filtering
    -g, --skip_gram       Flag to force Word2Vec to use skip-gram instead of CBOW

If you want to run the process in the background I recommend using nohup (don't forget the -u flag, otherwise it won't log properly):

`nohup python3 -u main.py [args] &`

### 2. Optuna Optimization

Script for running hyperparameter optimization of the Word2Vec model with Optuna.
Not a CLI, so if you need to change anything, do it in the code!
Note: We might switch to Wandb Sweep in the future, so this script will probably either be removed or changed. Creating a proper CLI is also in my plans.

## Notebooks

### 3. Topic Analysis

Jupyter notebook that allows you to visually inspect the topic in the corpus with pyLDAvis.
