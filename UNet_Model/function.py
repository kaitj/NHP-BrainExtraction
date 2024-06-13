"""Module for functions associated with processing volumes / datasets."""

from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
from dataset import BlockDataset, VolumeDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader


def write_nifti(
    data: np.ndarray, affine: np.ndarray, shape: tuple[int, ...], out_path: str
) -> None:
    """Write data to Nifti."""
    data = data[: shape[0], : shape[1], : shape[2]]
    img = nib.Nifti1Image(data, affine)
    img.to_filename(out_path)


def estimate_dice(gt_msk: np.ndarray, prt_msk: np.ndarray) -> float:
    """Compute Dice score given two binary masks."""
    intersection = gt_msk * prt_msk
    union = float(gt_msk.sum() + prt_msk.sum())
    return 2 * float(intersection.sum()) / union


def extract_large_comp(prt_msk: np.ndarray) -> np.ndarray:
    """Extract largest component from mask."""
    labs, _ = scipy.ndimage.label(prt_msk)
    c_size = np.bincount(labs.reshape(-1))
    c_size[0] = 0
    prt_msk = labs == c_size.argmax()
    return prt_msk


def fill_holes(prt_msk: np.ndarray) -> np.ndarray:
    """File holes in given mask."""
    non_prt_msk = extract_large_comp(prt_msk == 0)
    prt_msk_filled = non_prt_msk == 0
    return prt_msk_filled


def erosion_dilation(
    prt_msk: np.ndarray,
    structure: np.ndarray = scipy.ndimage.generate_binary_structure(3, 1),
    iterations: int = 1,
) -> np.ndarray:
    """Perform morphological closing on given mask."""
    prt_msk_eroded = scipy.ndimage.binary_erosion(
        prt_msk, structure=structure, iterations=iterations
    ).astype(prt_msk.dtype)
    prt_msk_eroded = extract_large_comp(prt_msk_eroded)
    prt_msk_dilated = scipy.ndimage.binary_dilation(
        prt_msk_eroded, structure=structure, iterations=iterations
    ).astype(prt_msk.dtype)
    return prt_msk_dilated


def setup_device(module: nn.Module) -> None:
    """Setup CPU/GPU processing."""
    use_gpu = torch.cuda.is_available()
    module_on_gpu = next(module.parameters()).is_cuda
    if use_gpu and not module_on_gpu:
        module.cuda()
    elif not use_gpu and module_on_gpu:
        module.cpu()


def predict_volumes(
    module: nn.Module,
    rimg_in: str | None = None,
    cimg_in: str | None = None,
    bmsk_in: str | None = None,
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
    setup_device(module)
    dice_dict = {} if save_dice else None

    if rimg_in is None and cimg_in is None:
        raise ValueError("One of 'rimg_in' or 'cimg_in' is not provided.")

    volume_dataset = VolumeDataset(rimg_in=rimg_in, cimg_in=cimg_in, bmsk_in=bmsk_in)
    volume_loader = DataLoader(dataset=volume_dataset, batch_size=1)

    for vol in volume_loader:
        block_dataset, ptype, cimg, bfld, bmsk = create_block_dataset(
            vol, num_slice, rescale_dim
        )
        process_volume(
            module,
            block_dataset,
            ptype,
            cimg,
            bfld,
            bmsk,
            volume_dataset,
            suffix,
            ed_iter,
            rescale_dim,
            save_dice,
            save_nii,
            nii_outdir,
            verbose,
            dice_dict,
        )

    return dice_dict if save_dice else None


def create_block_dataset(
    vol: list[torch.Tensor], num_slice: int, rescale_dim: int
) -> tuple[BlockDataset, int, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Helper function to create block dataset."""
    if len(vol) == 1:  # just img
        ptype = 1  # Predict
        cimg = vol
        bfld = None
        bmsk = None
    elif len(vol) == 2:  # img & msk
        ptype = 2  # image test
        cimg = vol[0]
        bfld = None
        bmsk = vol[1]
    elif len(vol) == 3:  # img bias_field & msk
        ptype = 3  # image bias correction test
        cimg = vol[0]
        bfld = vol[1]
        bmsk = vol[2]
    else:
        raise ValueError("Invalid Volume Dataset!")

    block_dataset = BlockDataset(
        rimg=cimg, bfld=bfld, bmsk=bmsk, num_slice=num_slice, rescale_dim=rescale_dim
    )
    return block_dataset, ptype, cimg, bfld, bmsk


def process_volume(
    module: nn.Module,
    block_dataset: BlockDataset,
    ptype: int,
    cimg: torch.Tensor,
    bfld: torch.Tensor | None,
    bmsk: torch.Tensor | None,
    volume_dataset: VolumeDataset,
    suffix: str,
    ed_iter: int,
    rescale_dim: int,
    save_dice: bool,
    save_nii: bool,
    nii_outdir: str | None,
    verbose: bool,
    dice_dict: dict[str, float] | None,
) -> None:
    """Helper function to process volumes."""
    rescale_shape = block_dataset.rescale_shape
    raw_shape = block_dataset.raw_shape

    for od in range(3):
        pr_bmsk = process_slice(od, module, block_dataset, ptype, rescale_dim)
        pr_bmsk = resize_predicted_mask(pr_bmsk, rescale_shape, raw_shape)
        pr_bmsk_final = postprocess_mask(pr_bmsk, ed_iter)

        if bmsk is not None:
            bmsk_np = bmsk.data[0].numpy()
            dice = estimate_dice(bmsk_np, pr_bmsk_final)
            if verbose:
                print(dice)
            if save_dice:
                t1w_name = get_volume_name(volume_dataset)
                dice_dict[t1w_name] = dice

        if save_nii:
            save_predicted_mask_as_nifti(
                pr_bmsk_final, volume_dataset, suffix, nii_outdir
            )


def process_slice(
    od: int,
    module: nn.Module,
    block_dataset: BlockDataset,
    ptype: int,
    rescale_dim: int,
) -> torch.Tensor:
    """Helper function to process slices."""
    backard_ind = np.arange(3)
    backard_ind = np.insert(np.delete(backard_ind, 0), od, 0)

    block_data, slice_list, slice_weight = block_dataset.get_one_directory(axis=od)
    pr_bmsk = (
        torch.zeros([len(slice_weight), rescale_dim, rescale_dim]).cuda()
        if torch.cuda.is_available()
        else torch.zeros([len(slice_weight), rescale_dim, rescale_dim])
    )

    for idx, ind in enumerate(slice_list):
        rimg_blk, _, _ = get_block_data(ptype, block_data, idx)
        pr_bmsk_blk = module(torch.unsqueeze(Variable(rimg_blk), 0))
        pr_bmsk[ind[1], :, :] = pr_bmsk_blk.data[0][1, :, :]

    pr_bmsk = pr_bmsk.cpu() if torch.cuda.is_available() else pr_bmsk
    pr_bmsk = pr_bmsk.permute(backard_ind[0], backard_ind[1], backard_ind[2])
    return pr_bmsk


def get_block_data(
    ptype: int,
    block_data: list[torch.Tensor],
    idx: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Function to get data from block."""
    torch_cuda = torch.cuda.is_available()
    if ptype == 1:
        rimg_blk = block_data[idx].cuda() if torch_cuda else block_data[idx]
        return rimg_blk, None, None
    elif ptype == 2:
        rimg_blk, bmsk_blk = block_data[idx]
        rimg_blk = rimg_blk.cuda() if torch_cuda else rimg_blk
        bmsk_blk = bmsk_blk.cuda() if torch_cuda else bmsk_blk
        return rimg_blk, None, bmsk_blk
    else:
        rimg_blk, bfld_blk, bmsk_blk = block_data[idx]
        rimg_blk = rimg_blk.cuda() if torch_cuda else rimg_blk
        bfld_blk = bfld_blk.cuda() if torch_cuda else bfld_blk
        bmsk_blk = bmsk_blk.cuda() if torch_cuda else bmsk_blk
        return rimg_blk, bfld_blk, bmsk_blk


def resize_predicted_mask(
    pr_bmsk: torch.Tensor,
    rescale_shape: tuple[int, int, int],
    raw_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Reshape predicted mask."""
    pr_bmsk = pr_bmsk[: rescale_shape[0], : rescale_shape[1], : rescale_shape[2]]
    uns_pr_bmsk = nn.functional.interpolate(
        torch.unsqueeze(torch.unsqueeze(pr_bmsk, 0), 0),
        size=raw_shape,
        mode="trilinear",
        align_corners=False,
    )
    return torch.squeeze(uns_pr_bmsk)


def postprocess_mask(pr_bmsk: torch.Tensor, ed_iter: int) -> np.ndarray:
    """Perform postprocessing (e.g. morphological closing)."""
    pr_bmsk = pr_bmsk.numpy()
    pr_bmsk_final = extract_large_comp(pr_bmsk > 0.5)
    pr_bmsk_final = fill_holes(pr_bmsk_final)
    if ed_iter > 0:
        pr_bmsk_final = erosion_dilation(pr_bmsk_final, iterations=ed_iter)
    return pr_bmsk_final


def get_volume_name(volume_dataset: VolumeDataset) -> str:
    """Get name of current volume."""
    t1w_nii = volume_dataset.cur_cimg_nii
    t1w_path = t1w_nii.get_filename()
    t1w_dir, t1w_file = os.path.split(t1w_path)
    t1w_dir = t1w_dir if t1w_dir else os.curdir
    t1w_name = os.path.splitext(os.path.splitext(t1w_file)[0])[0]
    return t1w_name


def save_predicted_mask_as_nifti(
    pr_bmsk_final: np.ndarray,
    volume_dataset: VolumeDataset,
    suffix: str,
    nii_outdir: str | None,
) -> None:
    """Save predicted mask as Nifti."""
    t1w_nii = volume_dataset.cur_cimg_nii
    t1w_aff = t1w_nii.affine
    t1w_shape = t1w_nii.shape
    t1w_dir = os.path.dirname(t1w_nii.get_filename())

    if nii_outdir is None:
        nii_outdir = t1w_dir

    if not os.path.exists(nii_outdir):
        os.makedirs(nii_outdir)

    t1w_path = t1w_nii.get_filename()
    t1w_dir, t1w_file = os.path.split(t1w_path)
    t1w_dir = t1w_dir if t1w_dir else os.curdir
    t1w_name = os.path.splitext(os.path.splitext(t1w_file)[0])[0]

    out_path = os.path.join(nii_outdir, f"{t1w_name}_{suffix}.nii.gz")
    write_nifti(np.array(pr_bmsk_final, dtype=np.float32), t1w_aff, t1w_shape, out_path)


# Unit test
# if __name__ == "__main__":
#     nifile = sys.argv[1]
#     nii = nib.load(nifile)
#     nii_data = nii.get_data()
#     ed_data = erosion_dilation(nii_data, iterations=1)
#     write_nifti(ed_data, nii.affine, ed_data.shape, "test.nii.gz")
