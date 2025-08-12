# -*- coding: utf-8 -*-
# @Time    : 2020/4/25 16:01
# @Author  : Hui Wang

import numpy as np
import random
import torch
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler

import os
import argparse

from dataset_s3rec import PretrainDataset
from trainers import PretrainTrainer
from model_s3rec import S3RecModel


from utils import get_user_seqs_long, get_item2attribute_json, check_path, set_seed, parse_user_seqs, parse_item_attr, parse_item_set, print_memory

def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    # parser.add_argument('--data_name', default='Beauty', type=str)

    # model args
    parser.add_argument("--model_name", default='Pretrain', type=str)

    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    # pre train args
    parser.add_argument("--pre_epochs", type=int, default=300, help="number of pre_train epochs")
    parser.add_argument("--pre_batch_size", type=int, default=100)

    parser.add_argument("--mask_p", type=float, default=0.2, help="mask probability")
    parser.add_argument("--aap_weight", type=float, default=0.2, help="aap loss weight")
    parser.add_argument("--mip_weight", type=float, default=1.0, help="mip loss weight")
    parser.add_argument("--map_weight", type=float, default=1.0, help="map loss weight")
    parser.add_argument("--sp_weight", type=float, default=0.5, help="sp loss weight")

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    parser.add_argument("--local_test", type=str, default=False, help="if local test, use the fixed dataset path")


    args = parser.parse_args()

    # set_seed(args.seed)
    # check_path(args.output_dir)

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    if args.local_test:
        args.data_dir = "F:\\Work\\202508_TencentAd\\TencentGR_1k\\TencentGR_1k\\"
        log_file = open(Path("E:\\zengyue\\QingyunCandidates\\models\\S3Rec\\output", 'train.log'), 'w')
    else:
        args.data_dir = os.environ.get('TRAIN_DATA_PATH')
        Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
        Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
        log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    args.data_file = Path(args.data_dir, 'seq.jsonl')
    # concat all user_seq get a long sequence, from which sample neg segment for SP
    print('*'*10, '\n', "Starting parse item2attr file")
    max_item, attribute_size, item2attribute = parse_item_set(args.data_dir)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1
    # save model args
    print(args)
    args.log_file = log_file
    args.log_file.write(str(args) + '\n')
    args.log_file.flush()

    args.item2attribute = item2attribute
    print_memory("MEMORY BEFORE 'model = S3RecModel(args=args)'")
    model = S3RecModel(args=args)
    trainer = PretrainTrainer(model, None, None, None, args)
    print_memory("MEMORY BEFORE 'for epoch in range(args.pre_epochs):'")
    for epoch in range(args.pre_epochs):

        pretrain_dataset = PretrainDataset(args, args.mask_p, args.mask_id, args.item_size, args.attribute_size, item2attribute, args.data_dir)
        pretrain_sampler = RandomSampler(pretrain_dataset)
        pretrain_dataloader = DataLoader(pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size)

        trainer.pretrain(epoch, pretrain_dataloader)
        

        if (epoch) % 10 == 0:
            if args.local_test:
                checkpoint_path = Path("E:\\zengyue\\QingyunCandidates\\models\\S3Rec\\output", f"global_step{epoch}")
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                trainer.save(Path(checkpoint_path, "ckpt.pt"))
            else:
                checkpoint_path = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{epoch}")
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                trainer.save(Path(checkpoint_path, "ckpt.pt"))


main()