from run_pipeline import *  
import argparse 
import os 
from config import ExperimentConfig

def main():
    parser = argparse.ArgumentParser(description="Models: [('ic-gan', 'gan'), ('vq-vae', vae'), 'diff', 'reg', ('eeg_ae', 'img_ae')]")
    parser.add_argument('-n', '--model_name', default = 'vae', type = str)
    parser.add_argument('-e', '--num_epochs', default = 500, type = int)
    parser.add_argument('-m', '--mode', default = 'train', type = str)
    parser.add_argument('-bs', '--batch_size', default = 32, type = int)
    args = vars(parser.parse_args())
    config = ExperimentConfig(**args)
    run_pipeline(config)

if __name__== '__main__':
    main()
