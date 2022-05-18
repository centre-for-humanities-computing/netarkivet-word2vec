# Netarkivet-word2vec

Code for training static word and document embeddings on the Danish web archive, Netarkivet.

## Usage

- Make sure to add a wandb.key file in the current directory containing the key to Weights and Biases
- If you intend to log the results to a different wandb project, make sure to modify the main script (This might change in the future, it will probably either read a file or use command line arguments)
- If the script fails to connect to the wandb project specified in the source code, it will log to stdout, so no worries
- make sure to run **setup.sh** before you run the script, it will install dependencies and log into wandb

# TODO: Rewrite this

#### You can run the code in the following fashion:

`python3 main.py --data_path [path to the data] --save_path [path to save the model to] --non_duplicates_path [path to the non-duplicate lists]`

If you want to run the process in the background I recommend using nohup:

`nohup python3 -u main.py [args] &`

To see all command line arguments consult the documentation or run:

`python3 main.py -h`
