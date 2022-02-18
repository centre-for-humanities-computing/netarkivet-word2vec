#!/bin/bash
KEY=$( cat "wandb.key" )
pip install -r requirements.txt
wandb login $KEY