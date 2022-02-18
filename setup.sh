#!/bin/bash
KEY=$( cat "wandb.key" )
pip install gensim wandb argparse
wandb login $KEY