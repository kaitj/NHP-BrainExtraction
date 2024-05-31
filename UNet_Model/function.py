"""Module for functions associated with processing volumes / datasets."""

import os

import nibabel as nib
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
from dataset import BlockDataset, VolumeDataset
from numpy.typing import NDArray
from torch.autograd import Variable
from torch.utils.data import DataLoader


def write_nifti(data: NDArray, affine: NDArray, shape: NDArray, out_path: str) -> None:
    """Write data to Nifti."""
    data = data[: shape[0], : shape[1], : shape[2]]
    img = nib.Nifti1Image(data, affine=affine)
    img.to_filename(out_path)


def estimate_dice(gt_mask: NDArray, prt_mask: NDArray) -> float:
    """Compute Dice score given two binary masks."""
    intersection = gt_mask * prt_mask
    union = float(gt_mask.sum() + prt_mask.sum())
    return 2 * float(intersection.sum()) / union


def extract_large_comp(prt_mask: NDArray) -> NDArray:
    """Extract largest component from mask."""
    labels, _ = scipy.ndimage.label(prt_mask)
    comp_size = np.bincount(labels.reshape(-1))
    comp_size[0] = 0
    prt_mask = labels == comp_size.argmax()
    return prt_mask


def fill_holes(prt_mask: NDArray) -> NDArray:
    """Fill holes in a given mask."""
    non_prt_mask = prt_mask == 0
    non_prt_mask = extract_large_comp(non_prt_mask)
    prt_mask_filled = non_prt_mask == 0
    return prt_mask_filled


def erosion_dilation(
    prt_mask: NDArray,
    structure: NDArray = scipy.ndimage.generate_binary_structure(3, 1),
    iterations: int = 1,
) -> NDArray:
    """Perform morphological closing on an image."""
    # Erosion
    prt_mask_eroded = scipy.ndimage.binary_erosion(
        prt_mask, structure=structure, iterations=iterations
    ).astype(prt_mask.dtype)

    # Extract Largest Component
    prt_mask_eroded = extract_large_comp(prt_mask_eroded)

    # Dilation
    prt_mask_dilated = scipy.ndimage.binary_dilation(
        prt_mask_eroded, structure=structure, iterations=iterations
    ).astype(prt_mask.dtype)

    return prt_mask_dilated


def predict_volumes(
    model: nn.Module,
    raw_img_in: str | None = None,
    corrected_img_in: str | None = None,
    brainmask_in: str | None = None,
    suffix: str = "pre_mask",
    ed_iter: int = 0,
    save_dice: bool = False,
    save_nii: bool = False,
    nii_outdir: str | None = None,
    verbose: bool = False,
    rescale_dim: int = 256,
    num_slice: int = 3,
) -> dict[str, float] | None:
    """Predict volume, optionally returning dice scores."""
    # Check if GPU is available and if what model is currently run on.
    use_gpu = torch.cuda.is_available()
    model_on_gpu = next(model.parameters()).is_cuda
    if use_gpu and not model_on_gpu:
        model.cuda()
    elif not use_gpu and model_on_gpu:
        model.cpu()

    if not raw_img_in and not corrected_img_in:
        raise ValueError("Provide one of 'raw_img_in' or 'corrected_img_in'")

    # Initialize dictionary to save dice scores
    dice_dict = {}

    volume_dataset = VolumeDataset(
        raw_img_in=raw_img_in,
        corrected_img_in=corrected_img_in,
        brainmask_in=brainmask_in,
    )
    volume_loader = DataLoader(dataset=volume_dataset, batch_size=1)

    for vol in volume_loader:
        if len(vol) == 1:  # just img
            ptype = 1
            corrected_img = vol[0]
            brainmask = None
        elif len(vol) == 2:
            ptype = 2
            corrected_img, brainmask = vol
        elif len(vol) == 3:
            ptype = 3
            corrected_img, bfld, brainmask = vol
        else:
            raise ValueError("Invalid volume dataset encountered!")

        block_dataset = BlockDataset(
            raw_img=corrected_img,
            bfld=bfld if ptype == 3 else None,
            brainmask=brainmask,
            num_slice=num_slice,
            rescale_dim=rescale_dim,
        )

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
                block = block_data[i]

                # Get block data
                raw_img_blk = block[0]
                if ptype == 2:
                    brainmask_blk = block[1]
                elif ptype == 3:
                    bfd_blk, brainmask_blk = block[1:]

                # Set gpu
                if use_gpu:
                    raw_img_blk = raw_img_blk.cuda()
                    if ptype == 2:
                        brainmask_blk = brainmask_blk.cuda()
                    elif ptype == 3:
                        bfld_blk = bfld_blk.cuda()  # noqa: F821, F841
                        brainmask_blk = brainmask_blk.cuda()

                pr_brainmask_blk = model(torch.unsqueeze(Variable(raw_img_blk), 0))
                pr_brainmask[ind[1], :, :] = pr_brainmask_blk.data[0][1, :, :]

            # Set back to cpu for brainmask if gpu was used
            if use_gpu:
                pr_brainmask = pr_brainmask.cpu()

            pr_brainmask = pr_brainmask.permute(
                backard_ind[0], backard_ind[1], backard_ind[2]
            )
            pr_brainmask = pr_brainmask[
                : rescale_shape[0], : rescale_shape[1], : rescale_shape[2]
            ]
            unsqueezed_pr_brainmask = torch.unsqueeze(pr_brainmask, 0)
            unsqueezed_pr_brainmask = torch.unsqueeze(unsqueezed_pr_brainmask, 0)
            unsqueezed_pr_brainmask = nn.functional.interpolate(
                unsqueezed_pr_brainmask,
                size=raw_shape,
                mode="trilinear",
                align_corners=False,
            )
            pr_brainmask = torch.squeeze(unsqueezed_pr_brainmask)

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

        # Convert to numpy if a Tensor and estimate Dice
        if isinstance(brainmask, torch.Tensor):
            brainmask = brainmask.data[0].numpy()
        dice = estimate_dice(brainmask, pr_brainmask_final)
        if verbose:
            print(dice)

        t1w_nii = volume_dataset.cur_corrected_img_nii
        t1w_path = t1w_nii.get_filename()
        t1w_dir, t1w_file = os.path.split(t1w_path)
        if not t1w_dir:
            t1w_dir = os.curdir
        t1w_name = t1w_file.split(".")[0]

        if save_nii:
            t1w_affine = t1w_nii.affine
            t1w_shape = t1w_nii.shape

            if not nii_outdir:
                nii_outdir = t1w_dir

            if not os.path.exists(nii_outdir):
                os.makedirs(nii_outdir, exist_ok=True)

            out_path = f"{nii_outdir}/{t1w_name}_{suffix}.nii.gz"
            write_nifti(
                data=np.array(pr_brainmask_final, dtype=np.float32),
                affine=t1w_affine,
                shape=t1w_shape,
                out_path=out_path,
            )

        if save_dice:
            dice_dict[t1w_name] = dice

    return dice_dict


# Unit test
# if __name__ == "__main__":
#     nifile = sys.argv[1]
#     nii = nib.load(nifile)
#     nii_data = nii.get_data()
#     ed_data = erosion_dilation(nii_data, iterations=1)
#     write_nifti(ed_data, nii.affine, ed_data.shape, "test.nii.gz")
