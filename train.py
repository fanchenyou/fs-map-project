import os
import yaml
import time
import shutil
import torch
import random
import pickle
import argparse
import numpy as np
import datetime
from torch.utils import data
from torch.optim.lr_scheduler import MultiStepLR

from loader.airsim_fsl_dataset import airsim_fsl_dataset
from loader.samplers import CategoriesSampler
from tools.utils import get_logger, load_model
from models.agent import *
from trainer.trainer import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="config")
    # about path
    parser.add_argument('--ph', type=int, default=1, choices=[0, 1], help='pre|train|test')
    parser.add_argument('--pretrain_dir', type=str, default='', help='path of models')
    parser.add_argument('--is_seg', type=int, default=0, choices=[0, 1], help='classification|segmentation')
    # parser.add_argument('--model_dir', type=str, default='', help='path of models')

    # about training
    parser.add_argument("--config", type=str, default="configs/mrms_fsl.yml", help="Configuration file to use", )
    parser.add_argument("--gpu", type=str, default="0", help="Used GPUs")
    parser.add_argument('--bsize', type=int, default=2, help='batch size of tasks')
    parser.add_argument('--val_frequency', type=int, default=50, help="Validate every 50 episodes")
    # about task
    parser.add_argument('--way', type=int, default=3)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=9, help='number of query image per class')
    parser.add_argument('--val_episode', type=int, default=1000, help='number of validation episode')
    parser.add_argument('--test_episode', type=int, default=2000, help='number of testing episodes after training')
    # solver
    parser.add_argument('--solver', type=str, default='sinkhorn', choices=['opencv', 'qpth', 'sinkhorn'])
    # recurrent
    parser.add_argument('--loop', type=int, default=0)
    parser.add_argument('--miter', type=int, default=10)
    # SFC
    parser.add_argument('--sfc_lr', type=float, default=0.05, help='learning rate of SFC')
    parser.add_argument('--sfc_wd', type=float, default=0, help='weight decay for SFC weight')
    parser.add_argument('--sfc_update_step', type=int, default=10, help='number of updating step of SFC')
    parser.add_argument('--sfc_bs', type=int, default=4, help='batch size for finetuning sfc')
    # Attention
    parser.add_argument('--head', type=int, default=1)
    # Use weight max to obtain the source and dst node weight
    parser.add_argument('--max_weight', type=int, default=0)

    args = parser.parse_args()

    # Set the gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    data_splits = ['val_split', 'test_split']

    if args.ph == 0:
        mode = 'pre_train'
        args.way = 6 - 1  # sample 10 classes (exclude bg)
        args.n_agent = args.way
    elif args.ph == 1:
        mode = 'meta'
        args.way = 3
        args.n_agent = args.way

    # we assume n_agent==n_way

    args.pre_train = (args.ph == 0)
    args.mode = mode

    # run_id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_id = datetime.datetime.now().strftime('%m-%d-%H-%M')

    prefix = 'class'
    if args.is_seg:
        prefix = 'seg'

    if args.ph > 0:
        logdir = os.path.join("results", prefix, mode, run_id, 'lp_%02d' % (args.loop,))
    else:
        assert args.loop == 0
        logdir = os.path.join("results", prefix, mode, run_id)

    os.makedirs(logdir, exist_ok=True)
    print("Model dir {}".format(logdir))
    shutil.copy(args.config, logdir)

    # ============= Training =============
    # logger
    logger = get_logger(logdir)
    logger.info("Begin")

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_path = cfg["data"]["path"]

    with open('configs/split_save_files.pkl', 'rb') as f:
        split_data_files = pickle.load(f)

    # print(split_data_files)
    b_size = cfg["training"]["batch_size"]
    n_worker = 4 if torch.cuda.is_available() else 1

    assert args.head == 1

    # datasets
    train_set = airsim_fsl_dataset(
        split_data_files['train'],
        is_transform=True,
        split=cfg["data"]["train_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=None
    )

    train_sampler = CategoriesSampler(train_set.all_labels, train_set.stat,
                                      args.val_frequency * args.bsize, args.way, args.shot + args.query)

    val_set = airsim_fsl_dataset(
        split_data_files['val'],
        is_transform=True,
        split=cfg["data"]["val_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
    )
    if args.loop >= 0:
        args.val_episode = int(args.val_episode * 0.6)
    val_sampler = CategoriesSampler(val_set.all_labels, val_set.stat,
                                    args.val_episode, args.way, args.shot + args.query)

    # n_worker = 1  # --------------------- remove this after debugging
    trainloader = data.DataLoader(train_set, batch_sampler=train_sampler, num_workers=n_worker, pin_memory=True)
    valloader = data.DataLoader(val_set, batch_sampler=val_sampler, num_workers=n_worker, pin_memory=True)

    print('fix val set for all epochs')
    val_sampler.mode = 'probe'
    val_set.mode = 'dummy'
    for x in valloader:
        pass
    val_sampler.mode = 'fix'
    val_set.mode = 'norm'
    print('fixed val set has %d batches' % (len(val_sampler.fixed_batches),))

    ########################################
    # Setup Model in models/agent.py
    ########################################

    assert args.solver == 'sinkhorn'

    n_classes = 11
    in_channels = 3
    if cfg["model"]["arch"] == 'MIMOcom':
        if args.shot == 1:
            model = MIMOcom(n_classes=n_classes, n_way=args.way, n_shot=args.shot, n_query=args.query,
                            in_channels=in_channels, mode=mode,
                            solver=args.solver, image_size=cfg["data"]["img_rows"],
                            query_size=cfg["model"]["query_size"], key_size=cfg["model"]["key_size"],
                            is_seg=args.is_seg, miter=args.miter, n_head=args.head, max_weight=args.max_weight)

        else:
            model = MIMOcom(n_classes=n_classes, n_way=args.way, n_shot=args.shot, n_query=args.query,
                            in_channels=in_channels, mode=mode,
                            solver=args.solver, image_size=cfg["data"]["img_rows"],
                            query_size=cfg["model"]["query_size"], key_size=cfg["model"]["key_size"],
                            is_seg=args.is_seg, miter=args.miter,
                            sfc_lr=args.sfc_lr, sfc_wd=args.sfc_wd, sfc_update_step=args.sfc_update_step,
                            sfc_bs=args.sfc_bs, n_head=args.head, max_weight=args.max_weight)

    else:
        raise ValueError('Incorrect arch')

    model = model.to(device)

    # print(torch.isnan(model.Wg).any(),'---Wg11111')

    ##################
    # resume training
    ##################

    if len(args.pretrain_dir) > 0:
        model_path = os.path.join(args.pretrain_dir, 'best_model.pth')
        print('--------------------------')
        print('Load pre-trained model %s' % (model_path,))
        print('--------------------------')

        if args.ph == 1:  # for loading pre-trained model
            load_model(model, model_path, strict=False)
        else:  # for resume training
            load_model(model, model_path, strict=True)

    ##################
    # Setup optimizer
    ##################
    if args.solver == 'sinkhorn':
        lr = 1e-3
    else:
        lr = 1e-4

    milestones = [10000, 20000, 30000]
    if args.ph == 1:
        lr *= 0.1
        milestones = [2000, 6000, 10000]
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('Lr is %.5f' % (lr,))
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # Loss function -- segmentation cross_entropy
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    ######################
    # train
    ######################
    trainer = Trainer_MIMOcom(cfg, args, logdir, logger, model, loss_fn, trainloader, valloader, optimizer,
                              scheduler, device)

    trainer.train()
