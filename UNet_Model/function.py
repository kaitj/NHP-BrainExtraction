import argparse
import os
import pickle
import sys

import nibabel as nib
import numpy as np
import scipy.ndimage as snd
import torch
import torch.nn as nn
from dataset import BlockDataset, VolumeDataset
from model import UNet2d
from torch.autograd import Variable
from torch.utils.data import DataLoader


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        self.exit(2)


def write_nifti(data, aff, shape, out_path):
    data = data[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    img = nib.Nifti1Image(data, aff)
    img.to_filename(out_path)


def estimate_dice(gt_msk, prt_msk):
    intersection = gt_msk * prt_msk
    dice = 2 * float(intersection.sum()) / float(gt_msk.sum() + prt_msk.sum())

    return dice


def extract_large_comp(prt_msk):
    labs, num_lab = snd.label(prt_msk)
    c_size = np.bincount(labs.reshape(-1))
    c_size[0] = 0
    max_ind = c_size.argmax()
    prt_msk = labs == max_ind

    return prt_msk


def fill_holes(prt_msk):
    non_prt_msk = prt_msk == 0
    non_prt_msk = extract_large_comp(non_prt_msk)
    prt_msk_filled = non_prt_msk == 0

    return prt_msk_filled


def erosion_dilation(
    prt_msk, structure=snd.generate_binary_structure(3, 1), iterations=1
):
    # Erosion
    prt_msk_eroded = snd.binary_erosion(
        prt_msk, structure=structure, iterations=iterations
    ).astype(prt_msk.dtype)

    # Extract Largest Component
    prt_msk_eroded = extract_large_comp(prt_msk_eroded)

    # Dilation
    prt_msk_dilated = snd.binary_dilation(
        prt_msk_eroded, structure=structure, iterations=iterations
    ).astype(prt_msk.dtype)

    return prt_msk_dilated


def predict_volumes(
    model,
    raw_img_in=None,
    corrected_img_in=None,
    brainmask_in=None,
    suffix="pre_mask",
    ed_iter=0,
    save_dice=False,
    save_nii=False,
    nii_outdir=None,
    verbose=False,
    rescale_dim=256,
    num_slice=3,
):
    use_gpu = torch.cuda.is_available()
    model_on_gpu = next(model.parameters()).is_cuda
    use_bn = True
    if use_gpu:
        if not model_on_gpu:
            model.cuda()
    else:
        if model_on_gpu:
            model.cpu()

    NoneType = type(None)
    if isinstance(raw_img_in, NoneType) and isinstance(corrected_img_in, NoneType):
        print("Input raw_img_in or corrected_img_in")
        sys.exit(1)

    if save_dice:
        dice_dict = dict()

    volume_dataset = VolumeDataset(
        raw_img_in=raw_img_in,
        corrected_img_in=corrected_img_in,
        brainmask_in=brainmask_in,
    )
    volume_loader = DataLoader(dataset=volume_dataset, batch_size=1)

    for idx, vol in enumerate(volume_loader):
        if len(vol) == 1:  # just img
            ptype = 1  # Predict
            corrected_img = vol
            brainmask = None
            block_dataset = BlockDataset(
                raw_img=corrected_img,
                bfld=None,
                brainmask=None,
                num_slice=num_slice,
                rescale_dim=rescale_dim,
            )
        elif len(vol) == 2:  # img & msk
            ptype = 2  # image test
            corrected_img = vol[0]
            brainmask = vol[1]
            block_dataset = BlockDataset(
                raw_img=corrected_img,
                bfld=None,
                brainmask=brainmask,
                num_slice=num_slice,
                rescale_dim=rescale_dim,
            )
        elif len(vol == 3):  # img bias_field & msk
            ptype = 3  # image bias correction test
            corrected_img = vol[0]
            bfld = vol[1]
            brainmask = vol[2]
            block_dataset = BlockDataset(
                raw_img=corrected_img,
                bfld=bfld,
                brainmask=brainmask,
                num_slice=num_slice,
                rescale_dim=rescale_dim,
            )
        else:
            print("Invalid Volume Dataset!")
            sys.exit(2)

        rescale_shape = block_dataset.rescale_shape()
        raw_shape = block_dataset.raw_shape()

        for od in range(3):
            backard_ind = np.arange(3)
            backard_ind = np.insert(np.delete(backard_ind, 0), od, 0)

            block_data, slice_list, slice_weight = block_dataset.get_one_directory(
                axis=od
            )
            pr_brainmask = torch.zeros([len(slice_weight), rescale_dim, rescale_dim])
            if use_gpu:
                pr_brainmask = pr_brainmask.cuda()
            for i, ind in enumerate(slice_list):
                if ptype == 1:
                    raw_img_blk = block_data[i]
                    if use_gpu:
                        raw_img_blk = raw_img_blk.cuda()
                elif ptype == 2:
                    raw_img_blk, brainmask_blk = block_data[i]
                    if use_gpu:
                        raw_img_blk = raw_img_blk.cuda()
                        brainmask_blk = brainmask_blk.cuda()
                else:
                    raw_img_blk, bfld_blk, brainmask_blk = block_data[i]
                    if use_gpu:
                        raw_img_blk = raw_img_blk.cuda()
                        bfld_blk = bfld_blk.cuda()
                        brainmask_blk = brainmask_blk.cuda()
                pr_brainmask_blk = model(torch.unsqueeze(Variable(raw_img_blk), 0))
                pr_brainmask[ind[1], :, :] = pr_brainmask_blk.data[0][1, :, :]

            if use_gpu:
                pr_brainmask = pr_brainmask.cpu()

            pr_brainmask = pr_brainmask.permute(
                backard_ind[0], backard_ind[1], backard_ind[2]
            )
            pr_brainmask = pr_brainmask[
                : rescale_shape[0], : rescale_shape[1], : rescale_shape[2]
            ]
            uns_pr_brainmask = torch.unsqueeze(pr_brainmask, 0)
            uns_pr_brainmask = torch.unsqueeze(uns_pr_brainmask, 0)
            uns_pr_brainmask = nn.functional.interpolate(
                uns_pr_brainmask, size=raw_shape, mode="trilinear", align_corners=False
            )
            pr_brainmask = torch.squeeze(uns_pr_brainmask)

            if od == 0:
                pr_3_brainmask = torch.unsqueeze(pr_brainmask, 3)
            else:
                pr_3_brainmask = torch.cat(
                    (pr_3_brainmask, torch.unsqueeze(pr_brainmask, 3)), dim=3
                )

        pr_brainmask = pr_3_brainmask.mean(dim=3)

        pr_brainmask = pr_brainmask.numpy()
        pr_brainmask_final = extract_large_comp(pr_brainmask > 0.5)
        pr_brainmask_final = fill_holes(pr_brainmask_final)
        if ed_iter > 0:
            pr_brainmask_final = erosion_dilation(
                pr_brainmask_final, iterations=ed_iter
            )

        if isinstance(brainmask, torch.Tensor):
            brainmask = brainmask.data[0].numpy()
            dice = estimate_dice(brainmask, pr_brainmask_final)
            if verbose:
                print(dice)

        t1w_nii = volume_dataset..cur_corrected_img_nii()
        t1w_path = t1w_nii.get_filename()
        t1w_dir, t1w_file = os.path.split(t1w_path)
        if t1w_dir == "":
            t1w_dir = os.curdir
        t1w_name = os.path.splitext(t1w_file)[0]
        t1w_name = os.path.splitext(t1w_name)[0]

        if save_nii:
            t1w_aff = t1w_nii.affine
            t1w_shape = t1w_nii.shape

            if isinstance(nii_outdir, NoneType):
                nii_outdir = t1w_dir

            if not os.path.exists(nii_outdir):
                os.mkdir(nii_outdir)
            out_path = os.path.join(nii_outdir, t1w_name + "_" + suffix + ".nii.gz")
            write_nifti(
                np.array(pr_brainmask_final, dtype=np.float32),
                t1w_aff,
                t1w_shape,
                out_path,
            )

        if save_dice:
            dice_dict[t1w_name] = dice

    if save_dice:
        return dice_dict


# Unit test
if __name__ == "__main__":
    nifile = sys.argv[1]
    nii = nib.load(nifile)
    nii_data = nii.get_data()
    ed_data = erosion_dilation(nii_data, iterations=1)
    write_nifti(ed_data, nii.affine, ed_data.shape, "test.nii.gz")
