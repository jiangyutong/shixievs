#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pprint import pprint

__all__ = ['Options']

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--data_dir',       type=str, default='../data/', help='path to dataset')
        self.parser.add_argument('--exp',            type=str, default='train', help='ID of experiment')
        self.parser.add_argument('--ckpt',           type=str, default='checkpoint/', help='path to save checkpoint')
        self.parser.add_argument('--results',           type=str, default='results/', help='path to save results')
        self.parser.add_argument('--load',           type=str, default='../libs/evolution/best_model.pth.pt', help='path to load a pretrained checkpoint')
        self.parser.add_argument('--test',           dest='test', action='store_true', help='test')
        self.parser.add_argument('--resume',         dest='resume', action='store_true', help='resume to train')

        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--max_norm',       dest='max_norm', action='store_true', help='maxnorm constraint to weights')
        self.parser.add_argument('--linear_size',    type=int, default=1024, help='size of each model layer')
        self.parser.add_argument('--num_stage',      type=int, default=2, help='# layers in linear model')
        self.parser.add_argument('--weight_l2',     type=int, default=100, metavar='S',help='weight for l2 loss (default: 100)')
        self.parser.add_argument('--weight_kl',     type=int, default=10, metavar='S',help='weight for l2 loss (default: 10)')
        self.parser.add_argument('--alpha', type=float, default=0.5, metavar='N', help='weight for cvae loss versus gsnn loss')
        self.parser.add_argument('--numSamples', type=int, default=1, metavar='N', help='num samples to cvae')
        self.parser.add_argument('--numSamples_train', type=int, default=10, metavar='N', help='num of samples from CVAE at train time for backpropagation')
        self.parser.add_argument('--cvae_num_stack', type=int, default=2, metavar='N',help='num of residual blocks in enc/dec of CVAE')
        self.parser.add_argument('--cvaeSize', type=int, default=1024, metavar='N', help='model capacity of cvae')
        self.parser.add_argument('--latent_size', type=int, default=256, metavar='N', help='size of latent layer')
        self.parser.add_argument('--cond_size', type=int, default=768, metavar='N', help='size of cond layer')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--use_hg',         dest='use_hg', action='store_true', help='whether use 2d pose from hourglass')
        self.parser.add_argument('--lr',             type=float,  default=2.5e-4)
        self.parser.add_argument('--lr_decay',       type=int,    default=100000, help='# steps of lr decay')
        self.parser.add_argument('--lr_gamma',       type=float,  default=0.96)
        self.parser.add_argument('--epochs',         type=int,    default=200)
        self.parser.add_argument('--dropout',        type=float,  default=0.5, help='dropout probability, 1.0 to make no dropout')
        self.parser.add_argument('--train_batch',    type=int,    default=256)
        self.parser.add_argument('--test_batch',     type=int,    default=256)
        self.parser.add_argument('--job',            type=int,    default=8, help='# subprocesses to use for data loading')
        self.parser.add_argument('--no_max',         dest='max_norm', action='store_false', help='if use max_norm clip on grad')
        self.parser.add_argument('--max',            dest='max_norm', action='store_true', help='if use max_norm clip on grad')
        self.parser.set_defaults(max_norm=True)
        self.parser.add_argument('--procrustes',     dest='procrustes', action='store_true', help='use procrustes analysis at testing')

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        # do some pre-check
        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        results = os.path.join(self.opt.results, self.opt.exp)
        if not os.path.isdir(ckpt):
            os.makedirs(ckpt)
        if not os.path.isdir(results):
            os.makedirs(results)
        if self.opt.load:
            if not os.path.isfile(self.opt.load):
                print ("{} is not found".format(self.opt.load))
        self.opt.is_train = False if self.opt.test else True
        self.opt.ckpt = ckpt
        self.opt.results = results
        self._print()
        return self.opt