"""Module associated with Torch datasets."""

from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from nibabel.arrayproxy import ArrayProxy


def load_files(input_path: str | None) -> tuple[str | None, list[str] | None]:
    """Function to help load files."""
    if input_path is None:
        return None, None
    if os.path.isdir(input_path):
        files = sorted(os.listdir(input_path))
        return input_path, files
    elif os.path.isfile(input_path):
        dir_path, file_name = os.path.split(input_path)
        return dir_path, [file_name]
    else:
        raise IOError(f"Invalid input path: {input_path}")


class VolumeDataset(data.Dataset):
    """Representation of a dataset of volumes."""

    def __init__(
        self,
        rimg_in: str | None = None,
        cimg_in: str | None = None,
        bmsk_in: str | None = None,
        debug: bool = True,
    ) -> None:
        super(VolumeDataset, self).__init__()
        self.rimg_dir, self.rimg_files = load_files(rimg_in)
        self.cimg_dir, self.cimg_files = load_files(cimg_in)
        self.bmsk_dir, self.bmsk_files = load_files(bmsk_in)

        # Current image vars.
        self.cur_rimg_nii: nib.Nifti1Image | None = None
        self.cur_cimg_nii: nib.Nifti1Image | None = None
        self.cur_bmsk_nii: nib.Nifti1Image | None = None

        # Debug flag
        self.debug = debug

    def __len__(self) -> int:
        """Get number of corrected."""
        assert isinstance(self.cimg_files, list)
        return len(self.cimg_files)

    def __getitem__(self, index: int) -> None:
        """Get Tensor representation of images at index."""
        if self.debug:
            print(self.rimg_files[index] if self.rimg_files else "")
            print(self.cimg_files[index] if self.cimg_files else "")
            print(self.bmsk_files[index] if self.bmsk_files else "")

        out = []
        if self.rimg_files:
            rimg_nii = nib.load(os.path.join(self.rimg_dir, self.rimg_files[index]))
            rimg = self.normalize(rimg_nii.dataobj)
            rimg = torch.from_numpy(rimg)
            out.append(rimg)
            self.cur_rimg_nii = rimg_nii

        if self.cimg_files:
            cimg_nii = nib.load(os.path.join(self.cimg_dir, self.cimg_files[index]))
            cimg = self.normalize(cimg_nii.dataobj)
            cimg = torch.from_numpy(cimg)
            out.append(cimg)
            self.cur_cimg_nii = cimg_nii

        if "rimg" in locals() and "cimg" in locals():
            bfld = cimg / rimg
            bfld[torch.isnan(bfld)] = 1
            bfld[torch.isinf(bfld)] = 1
            bfld = torch.from_numpy(bfld)
            out.append(bfld)

        if self.bmsk_files:
            bmsk_nii = nib.load(os.path.join(self.bmsk_dir, self.bmsk_files[index]))
            bmsk = (np.array(bmsk_nii.dataobj) > 0).astype(np.int64)
            bmsk = torch.from_numpy(bmsk)
            out.append(bmsk)
            self.cur_bmsk_nii = bmsk_nii

        if len(out) == 1:
            out = out[0]
        else:
            out = tuple(out)
        return out

    @staticmethod
    def normalize(data: np.ndarray | ArrayProxy) -> np.ndarray:
        """Helper function to normalize volume."""
        data = np.array(data)
        return (data - data.min()) / (data.max() - data.min())


class BlockDataset(data.Dataset):
    """Representation of a block of images."""

    def __init__(
        self,
        rimg: torch.Tensor,
        bfld: torch.Tensor | None = None,
        bmsk: torch.Tensor | None = None,
        num_slice: int = 3,
        rescale_dim: int = 256,
    ) -> None:
        """Initializes the BlockDataset."""
        super(BlockDataset, self).__init__()

        if isinstance(bmsk, torch.Tensor) and rimg.shape != bmsk.shape:
            print("Invalid shape of image and brain mask.")
            return

        raw_shape = rimg.shape[1:]  # exclude batch dimension
        max_dim = max(raw_shape)
        rescale_factor = float(rescale_dim) / float(max_dim)

        self.rimg = self._rescale(rimg, rescale_factor)
        self.bfld = self._rescale(bfld, rescale_factor) if bfld is not None else None
        self.bmsk = (
            self._rescale(bmsk.float(), rescale_factor, mode="nearest").long()
            if bmsk is not None
            else None
        )

        self.num_slice = num_slice
        self.rescale_dim = rescale_dim
        self.rescale_factor = rescale_factor
        self.rescale_shape = self.rimg.shape[1:]  # exclude batch dimension
        self.raw_shape = raw_shape
        self.batch_size = self.rimg.shape[0]

        self.slist0 = [
            range(i, i + num_slice)
            for i in range(self.rescale_shape[0] - num_slice + 1)
        ]
        self.slist1 = [
            range(i, i + num_slice)
            for i in range(self.rescale_shape[1] - num_slice + 1)
        ]
        self.slist2 = [
            range(i, i + num_slice)
            for i in range(self.rescale_shape[2] - num_slice + 1)
        ]

        self.batch_len = len(self.slist0) + len(self.slist1) + len(self.slist2)

    def _rescale(
        self, tensor: torch.Tensor | None, factor: float, mode: str = "trilinear"
    ) -> torch.Tensor | None:
        """Rescales a tensor using the given factor."""
        if tensor is None:
            return
        tensor = tensor.unsqueeze(0)
        tensor = nn.functional.interpolate(
            tensor, scale_factor=factor, mode=mode, align_corners=False
        )
        return tensor.squeeze(0)

    def get_one_directory(
        self, axis: int = 0
    ) -> tuple[list[torch.Tensor], list[range], np.ndarray]:
        """Returns one directory of slices along the given axis."""
        if axis == 0:
            ind = range(0, len(self.slist0))
            slist = self.slist0
        elif axis == 1:
            ind = range(len(self.slist0), len(self.slist0) + len(self.slist1))
            slist = self.slist1
        elif axis == 2:
            ind = range(
                len(self.slist0) + len(self.slist1),
                len(self.slist0) + len(self.slist1) + len(self.slist2),
            )
            slist = self.slist2

        slice_weight = np.zeros(slist[-1][-1] + 1)
        for lst in slist:
            slice_weight[lst] += 1

        slice_data = [self.__getitem__(i) for i in ind]

        return slice_data, slist, slice_weight

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return self.batch_size * self.batch_len

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Fetches the item at the given index."""
        bind = int(index / self.batch_len)
        index = index % self.batch_len

        if index < len(self.slist0):
            sind = self.slist0[index]
            rimg_tmp = self.rimg[bind][sind, :, :]

            if self.bfld is not None:
                bfld_tmp = self.bfld[bind][sind, :, :]

            if self.bmsk is not None:
                bmsk_tmp = self.bmsk[bind][sind, :, :]
        elif index < len(self.slist1) + len(self.slist0):
            sind = self.slist1[index - len(self.slist0)]
            rimg_tmp = self.rimg[bind][:, sind, :].permute(1, 0, 2)

            if self.bfld is not None:
                bfld_tmp = self.bfld[bind][:, sind, :].permute(1, 0, 2)

            if self.bmsk is not None:
                bmsk_tmp = self.bmsk[bind][:, sind, :].permute(1, 0, 2)
        else:
            sind = self.slist2[index - len(self.slist0) - len(self.slist1)]
            rimg_tmp = self.rimg[bind][:, :, sind].permute(2, 0, 1)

            if self.bfld is not None:
                bfld_tmp = self.bfld[bind][:, :, sind].permute(2, 0, 1)

            if self.bmsk is not None:
                bmsk_tmp = self.bmsk[bind][:, :, sind].permute(2, 0, 1)

        extend_dim = self.rescale_dim
        slice_shape = rimg_tmp.shape[1:]

        rimg_blk = torch.zeros(
            (self.num_slice, extend_dim, extend_dim), dtype=torch.float32
        )
        rimg_blk[:, : slice_shape[0], : slice_shape[1]] = rimg_tmp

        if self.bfld is not None:
            bfld_blk = torch.ones(
                (self.num_slice, extend_dim, extend_dim), dtype=torch.float32
            )
            bfld_blk[:, : slice_shape[0], : slice_shape[1]] = bfld_tmp
            return rimg_blk, bfld_blk

        if self.bmsk is not None:
            bmsk_blk = torch.zeros(
                (self.num_slice, extend_dim, extend_dim), dtype=torch.long
            )
            bmsk_blk[:, : slice_shape[0], : slice_shape[1]] = bmsk_tmp
            return rimg_blk, bmsk_blk

        return rimg_blk


# if __name__ == "__main__":
#     volume_dataset = VolumeDataset(
#         rimg_in=None,
#         cimg_in="../site-ucdavis/TrainT1w",
#         bmsk_in="../site-ucdavis/TrainMask",
#     )
#     volume_loader = data.DataLoader(dataset=volume_dataset, batch_size=1, shuffle=True)
#     for i, (cimg, bmsk) in enumerate(volume_loader):
#         block_dataset = BlockDataset(
#             rimg=cimg, bfld=None, bmsk=bmsk, num_slice=3, rescale_dim=256
#         )
#         block_loader = data.DataLoader(
#             dataset=block_dataset, batch_size=20, shuffle=True
#         )
#         for j, (cimg_blk, bmsk_blk) in enumerate(block_loader):
#             print(bmsk_blk.shape)
