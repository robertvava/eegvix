from run_pipeline import *  
import argparse 
import os 
from config import ExperimentConfig

def main():

    
    parser = argparse.ArgumentParser(description="Models: [('ic-gan', 'gan'), ('vq-vae', vae'), 'diff']")
    parser.add_argument('-n', '--model_name', default = 'vae', type=str)
    parser.add_argument('-e', '--num_epochs', default = 50, type=int)
    args = vars(parser.parse_args())
    config = ExperimentConfig(act = 'train', **args)
    run_pipeline(config)

if __name__== '__main__':
    main()