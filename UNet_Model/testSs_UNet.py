#!/usr/bin/env python
import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from function import predict_volumes
from model import UNet2d


def create_parser() -> argparse.ArgumentParser:
    """Parser for testing a model."""
    parser = argparse.ArgumentParser(
        description="Testing model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-tet1w", "--test_t1w", type=str, required=True, help="Test T1w directory."
    )
    required.add_argument(
        "-temsk", "--test_msk", type=str, required=True, help="Test mask directory."
    )
    required.add_argument(
        "-out", "--out_dir", type=str, required=True, help="Output directory."
    )
    required.add_argument(
        "-model", "--test_model", type=str, required=True, help="Test model."
    )

    # Optional arguments
    optional = parser.add_argument_group("optional arguments")
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
        help="Rescale to number of voxels.",
    )

    return parser


def check_test_directory(in_dir: str) -> None:
    """Helper to check validity of provided directory path."""
    if not os.path.exists(in_dir):
        raise NotADirectoryError(
            f"{in_dir} is an invalid directory, please check again!"
        )


def main() -> None:
    """Main entry point for testing a model."""
    parser = create_parser()
    args = parser.parse_args()

    # Check inputs
    check_test_directory(args.test_msk)
    check_test_directory(args.test_t1w)
    if not os.path.exists(args.test_model):
        raise ValueError("Invalid test model, please check again!")

    # Start testing model
    print("Testing model".center(88, "="))

    train_model = UNet2d(
        dim_in=args.input_slice,
        num_conv_block=args.conv_block,
        kernel_root=args.kernel_root,
    )
    checkpoint = torch.load(args.test_model, map_location={"cuda:0": "cpu"})
    train_model.load_state_dict(checkpoint["state_dict"])

    model = nn.Sequential(train_model, nn.Softmax2d())

    dice_dict = predict_volumes(
        model,
        raw_img_in=None,
        corrected_img_in=args.test_t1w,
        brainmask_in=args.test_msk,
        rescale_dim=args.rescale_dim,
        save_nii=True,
        nii_outdir=args.out_dir,
        save_dice=True,
    )
    dice_array = np.array([val for val in dice_dict.values()])
    print("\t%.4f +/- %.4f" % (dice_array.mean(), dice_array.std()))
    with open(os.path.join(args.out_dir, "Dice.pkl"), "wb") as out_fpath:
        pickle.dump(dice_dict, out_fpath)


if __name__ == "__main__":
    main()
