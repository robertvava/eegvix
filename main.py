from run_pipeline import *  
import argparse 
import os 
from config import config

def main():

    parser = argparse.ArgumentParser(description="Models: [('ic-gan', 'gan'), ('vq-vae', vae'), 'diff']")
    parser.add_argument('-n', '--model_name', type=str)
    args = vars(parser.parse_args())
    model_config = config(act = 'train', **args)
    run_pipeline(model_config)


    
if __name__== '__main__':
    main()