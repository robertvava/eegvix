from run import run_pipeline
import argparse 
import os 

def main():

    parser = argparse.ArgumentParser(description="Models: [('ic-gan', 'gan'), ('vq-vae', vae'), 'diff']")
    parser.add_argument('-n', '--model_name', type=str)
    args = vars(parser.parse_args())

if __name__== '__main__':
    main()