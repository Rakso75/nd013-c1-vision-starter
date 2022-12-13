import argparse
import glob
import os
import random
#import shutil
import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # TODO: Implement function
    filenames = os.listdir("/home/workspace/data/training_and_validation")

    #Shuffle tf records list
    np.random.shuffle(filenames)
    
    #Train size: 90%, validation size: 10%
    train_size = int(0.9 * len(filenames))
    val_size = len(filenames) - train_size
    
    train_filenames, val_filenames = np.split(filenames, [train_size])
    
    source_folder = "/home/workspace/data/training_and_validation/"
    
    train_folder = "/home/workspace/data/train/"
    #Move training records
    for filename in train_filenames:
        shutil.move(source_folder+filename, train_folder+filename)
        
    val_folder = "/home/workspace/data/val/"
    #Move validation records
    for filename in val_filenames:
        shutil.move(source_folder+filename, val_folder+filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)