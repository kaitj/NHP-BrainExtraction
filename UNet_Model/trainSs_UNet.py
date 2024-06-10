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


def create_parser() -> argparse.ArgumentParser:
    """Parser for training a model."""
    parser = argparse.ArgumentParser(
        description="Training model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-trt1w", "--train_t1w", type=str, required=True, help="Train T1w directory."
    )
    required.add_argument(
        "-trmsk", "--train_msk", type=str, required=True, help="Train mask directory."
    )
    required.add_argument(
        "-out", "--out_dir", type=str, required=True, help="Output directory."
    )

    # Optional arguments
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "-vt1w", "--validate_t1w", type=str, help="Validation T1w directory."
    )
    optional.add_argument(
        "-vmsk", "--validate_msk", type=str, help="Validation mask directory."
    )
    optional.add_argument("-init", "--init_model", type=str, help="Initialize model.")
    optional.add_argument(
        "-slice",
        "--input_slice",
        type=int,
        default=3,
        help="Number of slices for model input.",
    )
    optional.add_argument(
        "-conv", "--conv_block", type=int, default=5, help="Number of UNet blocks."
    )
    optional.add_argument(
        "-kernel", "--kernel_root", type=int, default=16, help="Number of kernel roots."
    )
    optional.add_argument(
        "-rescale",
        "--rescale_dim",
        type=int,
        default=256,
        help="Rescale to number of models.",
    )
    optional.add_argument(
        "-epoch", "--num_epoch", type=int, default=40, help="Number of epochs."
    )
    optional.add_argument(
        "-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate."
    )


def check_train_directory(in_dir: str) -> None:
    """Helper to check validity of provided directory path."""
    if not os.path.exists(in_dir):
        raise NotADirectoryError(
            f"{in_dir} is an invalid directory, please check again!"
        )


def main() -> None:
    """Main entry point for training a model."""
    parser = create_parser()
    args = parser.parse_args()

    # Check inputs
    check_train_directory(args.train_msk)
    check_train_directory(args.train_t1w)

    use_validate = True
    if (
        not args.validate_msk
        or not os.path.exists(args.validate_msk)
        or not args.validate_t1w
        or not os.path.exists(args.validate_t1w)
    ):
        use_validate = False
        print("Not validating dataset.")
    use_gpu = torch.cuda.is_available()

    # Start training model
    print("Training model".center(88, "="))

    model = UNet2d(
        dim_in=args.input_slice,
        num_conv_block=args.conv_block,
        kernel_root=args.kernel_root,
    )

    if isinstance(args.init_model, str):
        if not os.path.exists(args.init_model):
            raise ValueError("Invalid initialization model, please check again!")
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

    os.makedirs(args.out_dir, exist_ok=True)

    # Init Dice and Loss Dict
    dl_dict = dict()
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
        dice_array = np.array([val for val in dice_dict.values()])
        dl_dict["origin_dice"] = dice_array
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
            dice_array = np.array([val for val in dice_dict.values()])
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
    dl_dict["dice"] = dice_list
    dl_dict["loss"] = loss_list
    with open(os.path.join(args.out_dir, "DiceAndLoss.pkl"), "wb") as handle:
        pickle.dump((dice_list, loss_list), handle)
