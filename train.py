'''
Train an Electra-based classifer on the IMDB dataset
'''

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from transformers import ElectraTokenizer
from data_prep import get_train, get_val
import sys
import os
import argparse
from tools import AverageMeter, accuracy, get_default_device
from models import SequenceClassifier

def train():
    '''
    Run one train epoch
    '''

def eval():
    '''
    Run evaluation
    '''

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('OUT', type=str, help='Specify output th file')
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=25, help="Specify epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.001, help="Specify learning rate")
    commandLineParser.add_argument('--sch', type=float, default=0.99, help="Specify scheduler rate")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")

    args = commandLineParser.parse_args()
    out_file = args.OUT
    batch_size = args.B
    epochs = args.epochs
    lr = args.lr
    sch = args.sch
    seed = args.seed

    torch.manual_seed(seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    device = get_default_device()

    # Load the data as tensors
    input_ids_train, mask_train, labels_train = get_train()
    input_ids_val, mask_val, labels_val = get_val()

    # Use dataloader to handle batches
    train_ds = TensorDataset(input_ids_train, mask_train, labels_train)
    val_ds = TensorDataset(input_ids_val, mask_val, labels_val)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Initialise classifier
    model = SequenceClassifier()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
