import os
import sys
import time
import glob
import numpy as np
from matplotlib import image
import torch
import utils
import logging
import argparse
import genotypes
from tqdm import tqdm
from datetime import datetime as dt

import torch.nn as nn
import torch.backends.cudnn as cudnn
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
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='weight/weights.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--epsilon', type=float, default=0.3, help='adversarial training epsilon')
parser.add_argument('--test_mode', type=str, default='natural', const='natural', nargs='?', choices=['adversarial', 'natural'])
args = parser.parse_args()

# Record of experiment date and time
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10


def main():
    # --- Setting ---
    # gpu is available
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    
    torch.cuda.set_device(args.gpu)
    # optimize in cuDNN
    cudnn.benchmark = True 
    cudnn.enabled = True

    # set the seed of random
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # genotype
    genotype = eval("genotypes.%s" % args.arch)
    # print(genotype)
    model = Network(CIFAR_CLASSES)
    model = model.cuda()
    utils.load(model, args.model_path)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    _, test_transform = utils._data_transforms_cifar10(args)
    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model.drop_path_prob = args.drop_path_prob

    now = dt.now()
    start = now.strftime('%p%I:%M:%S')

    test_acc, test_obj = infer(test_queue, model, criterion, start)
    logging.info('test_acc %f', test_acc)


def infer(test_queue, model, criterion, start):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    attack = FastGradientSignUntargeted(model, args.epsilon, _type='l2')

    with tqdm(total=len(test_queue),unit='batch') as progress_bar:
        progress_bar.set_description("(test) start: " + start)

        for step, (input, target) in enumerate(test_queue, 0): # input: image, target: label
            input = Variable(input, volatile=True).cuda()
            target = Variable(target, volatile=True).cuda(non_blocking=True)

            if args.test_mode == 'natural':
                logits, _ = model(input)

            elif args.test_mode == 'adversarial':
                adv_sample = attack.perturb(input, target, 'mean')
                logits, _ = model(adv_sample)

            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

            now = dt.now()
            progress_bar.set_postfix({"loss":loss.item(),"accuracy":top1.avg, "now":now.strftime('%p%I:%M:%S')})
            progress_bar.update(1)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()