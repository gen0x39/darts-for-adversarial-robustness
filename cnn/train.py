import os
import time
import glob

from matplotlib import image
import torch
import utils
import logging
import argparse
import genotypes
from tqdm import tqdm

import torch.nn as nn
import torchvision.datasets as dset

from torch.autograd import Variable
from model import NetworkCIFAR as Network

from attack import FastGradientSignUntargeted
import visualize

# --- Experiment settings ---
# Command line arguments
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=2, help='num of training epochs')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--epsilon', type=float, default=0.3, help='adversarial training epsilon')
parser.add_argument('--training_mode', type=str, default='natural', const='natural', nargs='?', choices=['adversarial', 'natural'])
args = parser.parse_args()

# Record of experiment date and time
args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S")) # Example(eval-EXP-20210914-021627)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

CIFAR_CLASSES = 10

def main():
    # --- Setting ---
    # gpu is available
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    
    torch.cuda.set_device(args.gpu)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # genotype
    genotype = eval("genotypes.%s" % args.arch)
    print(genotype)
    model = Network(CIFAR_CLASSES)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()   # loss function
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(        # optimizer
        model.parameters(), 
        lr=args.learning_rate, 
        momentum=args.momentum
        )

    # --- Loading Data ---
    # transform: preprocessing
    # train_data: loading dataset
    # train_queue: get batch from datasets
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    
    # scheduling learning rate (change learning rate step by step)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # --- Training ---
    for epoch in range(args.epochs):    # loop over the dataset multiple times
        train(epoch, train_queue, model, criterion, optimizer)      # 
        utils.save(model, os.path.join(args.save, 'weights.pt'))    # save model as pt file
    print('Finished Training')

def train(epoch, train_queue, model, criterion, optimizer):
    running_loss = 0.0
    attack = FastGradientSignUntargeted(model, args.epsilon, _type='l2')

    with tqdm(total=len(train_queue),unit='batch') as progress_bar:
        progress_bar.set_description(f"Epoch[{epoch}/{args.epochs}](training)")

        for step, (input, target) in enumerate(train_queue, 0): # input: image, target: label
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            optimizer.zero_grad()    # zero the parameter gradients

            # generate adversarial example
            if args.training_mode == 'adversarial':
                adv_sample = attack.perturb(input, target, 'mean')
                logits, logits_aux = model(adv_sample)

            elif args.training_mode == 'natural':
                logits, logits_aux = model(input)

            # forward + backward + optimize
            loss = criterion(logits, target)
            loss.backward()

            # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 2000))
                running_loss = 0.0

                if args.training_mode == 'adversarial':
                    visualize.save_adversarial_img(adv_sample, input, target, args.epsilon, step)
            
            progress_bar.update(1)
            
    return running_loss

if __name__ == '__main__':
    main()