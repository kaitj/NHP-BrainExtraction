#!/usr/bin/env python
import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from dataset import BlockDataset, VolumeDataset
from function import predict_volumes
from model import UNet2d
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

if __name__ == "__main__":
    NoneType = type(None)
    # Argument
    parser = argparse.ArgumentParser(
        description="Training Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    # Required Option
    required.add_argument(
        "-trt1w", "--train_t1w", type=str, required=True, help="Train T1w Directory"
    )
    required.add_argument(
        "-trmsk", "--train_msk", type=str, required=True, help="Train Mask Directory"
    )
    required.add_argument(
        "-out", "--out_dir", type=str, required=True, help="Output Directory"
    )
    # Optional Option
    optional.add_argument(
        "-vt1w", "--validate_t1w", type=str, help="Validation T1w Directory"
    )
    optional.add_argument(
        "-vmsk", "--validate_msk", type=str, help="Validation Mask Directory"
    )
    optional.add_argument("-init", "--init_model", type=str, help="Init Model")
    optional.add_argument(
        "-slice",
        "--input_slice",
        type=int,
        default=3,
        help="Number of Slice for Model Input",
    )
    optional.add_argument(
        "-conv", "--conv_block", type=int, default=5, help="Number of UNet Block"
    )
    optional.add_argument(
        "-rescale",
        "--rescale_dim",
        type=int,
        default=256,
        help="Number of the Root of Kernel",
    )
    optional.add_argument(
        "-kernel",
        "--kernel_root",
        type=int,
        default=16,
        help="Number of the Root of Kernel",
    )
    optional.add_argument(
        "-epoch", "--num_epoch", type=int, default=40, help="Number of Epoch"
    )
    optional.add_argument(
        "-lr", "--learning_rate", type=float, default=0.0001, help="Number of Epoch"
    )
    parser._action_groups.append(optional)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    print(
        "===================================Training Model==================================="
    )

    if not os.path.exists(args.train_msk) or not os.path.exists(args.train_t1w):
        print("Invalid train directory, please check again!")
        sys.exit(2)

    use_validate = True
    if (
        isinstance(args.validate_msk, NoneType)
        or isinstance(args.validate_t1w, NoneType)
        or not os.path.exists(args.validate_msk)
        or not os.path.exists(args.validate_t1w)
    ):
        use_validate = False
        print("NOTE: Do not use validate dataset.")

    use_gpu = torch.cuda.is_available()
    model = UNet2d(
        dim_in=args.input_slice,
        num_conv_block=args.conv_block,
        kernel_root=args.kernel_root,
    )
    if isinstance(args.init_model, str):
        if not os.path.exists(args.init_model):
            print("Invalid init model, please check again!")
            sys.exit(2)
        checkpoint = torch.load(args.init_model, map_location={"cuda:0": "cpu"})
        model.load_state_dict(checkpoint["state_dict"])

    if use_gpu:
        model.cuda()
        cudnn.benchmark = True

    # optimizer
    optimizerSs = optim.Adam(model.parameters(), lr=args.learning_rate)

    # loss function
    criterionSs = nn.CrossEntropyLoss()
    if use_gpu:
        criterionSs.cuda()

    volume_dataset = VolumeDataset(
        raw_img_in=None, corrected_img_in=args.train_t1w, brainmask_in=args.train_msk
    )
    volume_loader = DataLoader(
        dataset=volume_dataset, batch_size=1, shuffle=True, num_workers=0
    )

    blk_batch_size = 20

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # Init Dice and Loss Dict
    DL_Dict = dict()
    dice_list = list()
    loss_list = list()

    if use_validate:
        valid_model = nn.Sequential(model, nn.Softmax2d())
        dice_dict = predict_volumes(
            valid_model,
            raw_img_in=None,
            corrected_img_in=args.validate_t1w,
            brainmask_in=args.validate_msk,
            rescale_dim=args.rescale_dim,
            num_slice=args.input_slice,
            save_nii=False,
            save_dice=True,
        )
        dice_array = np.array([v for v in dice_dict.values()])
        DL_Dict["origin_dice"] = dice_array
        print("Origin Dice: %.4f +/- %.4f" % (dice_array.mean(), dice_array.std()))

    for epoch in range(0, args.num_epoch):
        lossSs_v = []
        print("Begin Epoch %d" % epoch)
        for i, (corrected_img, brainmask) in enumerate(volume_loader):
            block_dataset = BlockDataset(
                raw_img=corrected_img,
                bfld=None,
                brainmask=brainmask,
                num_slice=args.input_slice,
                rescale_dim=args.rescale_dim,
            )
            block_loader = DataLoader(
                dataset=block_dataset,
                batch_size=blk_batch_size,
                shuffle=True,
                num_workers=0,
            )
            for j, (corrected_img_blk, brainmask_blk) in enumerate(block_loader):
                brainmask_blk = brainmask_blk[:, 1, :, :]
                corrected_img_blk, brainmask_blk = (
                    Variable(corrected_img_blk),
                    Variable(brainmask_blk),
                )
                if use_gpu:
                    corrected_img_blk = corrected_img_blk.cuda()
                    brainmask_blk = brainmask_blk.cuda()
                pr_brainmask_blk = model(corrected_img_blk)

                # Loss Backward
                lossSs = criterionSs(pr_brainmask_blk, brainmask_blk)
                optimizerSs.zero_grad()
                lossSs.backward()
                optimizerSs.step()

                if use_gpu:
                    lossSs = lossSs.cpu()

                lossSs_v.append(lossSs.data.detach().numpy())

                print(
                    "\tEpoch:%.2d [%.3d/%.3d (%.4d/%.4d)]\tLoss Ss: %.6f"
                    % (
                        epoch,
                        i,
                        len(volume_loader.dataset) - 1,
                        j * blk_batch_size,
                        len(block_loader.dataset),
                        lossSs.data.detach(),
                    )
                )
        loss = np.array(lossSs_v).sum()

        if use_validate:
            valid_model = nn.Sequential(model, nn.Softmax2d())
            dice_dict = predict_volumes(
                valid_model,
                raw_img_in=None,
                corrected_img_in=args.validate_t1w,
                brainmask_in=args.validate_msk,
                save_dice=True,
            )
            dice_array = np.array([v for v in dice_dict.values()])
            dice_list.append(dice_array)
            print(
                "\tEpoch: %d; Dice: %.4f +/- %.4f; Loss: %.4f"
                % (epoch, dice_array.mean(), dice_array.std(), loss)
            )
        else:
            dice_array = []
            print("\tEpoch: %d; Loss: %.4f" % (epoch, loss))

        if (epoch) % 1 == 0:
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizerSs": optimizerSs.state_dict(),
                "lossSs": lossSs_v,
                "validate_dice": dice_array,
            }
            torch.save(
                checkpoint, os.path.join(args.out_dir, "model-%.2d-epoch" % (epoch))
            )
    DL_Dict["dice"] = dice_list
    DL_Dict["loss"] = loss_list
    with open(os.path.join(args.out_dir, "DiceAndLoss.pkl"), "wb") as handle:
        pickle.dump((dice_list, loss_list), handle)
