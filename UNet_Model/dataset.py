from __future__ import annotations

import os
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from nibabel.arrayproxy import ArrayProxy
from numpy.typing import NDArray


class VolumeDataset(data.Dataset):
    """Representation of a dataset of volumes."""

    def __init__(
        self,
        raw_img_in: str | None = None,
        corrected_img_in: str | None = None,
        brainmask_in: str | None = None,
        debug: bool = True,
    ) -> None:
        super(VolumeDataset, self).__init__()

        self.raw_img_dir, self.raw_img_files = self._set_img_attrs(raw_img_in)
        self.corrected_img_dir, self.corrected_img_files = self._set_img_attrs(
            corrected_img_in
        )
        self.brainmask_dir, self.brainmask_files = self._set_img_attrs(brainmask_in)

        # Current image variables
        self.cur_raw_img_nii = None
        self.cur_corrected_img_nii = None
        self.cur_brainmask_nii = None

        # Debug flag
        self.debug = debug

    def __len__(self) -> int:
        """Get number of corrected image files."""
        return len(self.corrected_img_files) if self.corrected_img_files else 0

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        """Get Tensor representation of images at index."""
        if self.debug:
            print(self.raw_img_files[idx] if self.raw_img_files else "")
            print(self.corrected_img_files[idx] if self.corrected_img_files else "")
            print(self.brainmask_files[idx] if self.brainmask_files else "")

        out: list[torch.Tensor] = []
        if self.raw_img_files:
            raw_img_nii = nib.load(
                os.path.join(self.raw_img_dir, self.raw_img_files[idx])
            )
            raw_img = self._normalize_img(raw_img_nii.dataobj)
            out.append(torch.from_numpy(raw_img))
            self.cur_raw_img_nii = raw_img_nii

        if self.corrected_img_files:
            corrected_img_nii = nib.load(
                os.path.join(self.corrected_img_dir, self.corrected_img_files[idx])
            )
            corrected_img = self._normalize_img(corrected_img_nii.dataobj)
            out.append(torch.from_numpy(corrected_img))
            self.cur_corrected_img_nii = corrected_img_nii

        if "raw_img" in locals() and "corrected_img" in locals():
            bfld = corrected_img / raw_img
            bfld[np.isnan(bfld)] = 1
            bfld[np.isinf(bfld)] = 1
            bfld = torch.from_numpy(bfld)
            out.append(blfd)

        if self.brainmask_files:
            brainmask_nii = nib.load(
                os.path.join(self.brainmask_dir, self.brainmask_files[idx])
            )
            brainmask = torch.from_numpy(brainmask_nii.dataobj)
            out.append(brainmask)
            self.cur_brainmask_nii = brainmask_nii

        return out[-1] if len(out) == 1 else tuple(out)

    def _set_img_attrs(
        self, img_in: str | None
    ) -> tuple[None, ...] | tuple[str, list[str]]:
        """Internal function to check and set image attributes."""
        if img_in is None:
            return None, None

        if isinstance(img_in, str):
            if os.path.isdir(img_in):
                return img_in, sorted(os.listdir(img_in))
            elif os.path.isfile(img_in):
                img_dir, img_file = os.path.split(img_in)
                return img_dir, [img_file]

        raise ValueError(f"Invalid {img_in}")

    def _normalize_img(self, dataobj: ArrayProxy | NDArray) -> NDArray:
        """Internal function to perform image normalization."""
        dataobj = np.array(dataobj)
        return (dataobj - dataobj.min()) / (dataobj.max() - dataobj.min())


class BlockDataset(data.Dataset):
    """Representation of a block dataset."""

    def __init__(
        self,
        raw_img: torch.Tensor | None = None,
        bfld: torch.Tensor | None = None,
        brainmask: torch.Tensor | None = None,
        num_slice: int = 3,
        rescale_dim: float | int = 256,
    ):
        super(BlockDataset, self).__init__()

        # Check masks align
        if brainmask and raw_img and brainmask.shape != raw_img.shape:
            raise ValueError("Shape mismatch between brainmask and raw_img")

        self.raw_shape = raw_img.data[0].shape
        self.rescale_factor = float(rescale_dim) / torch.tensor(self.raw_shape).max()
        self.num_slice = num_slice

        self.raw_img = self._nn_interpolate(
            raw_img,
            rescale_factor=rescale_factor,
            mode="trilinear",
            align_corners=False,
        )
        self.bfld = self._nn_interpolate(
            bfld,
            scale_factor=self.rescale_factor,
            mode="trilinear",
            align_corners=False,
        )
        self.brainmask = self._nn_interpolate(
            brainmask, scale_factor=self.rescale_factor, mode="nearest"
        )

        self.rescale_shape = self.raw_img.data[0].shape

        self.slist0 = self._create_slist(self.rescale_shape[0])
        self.slist1 = self._create_slist(self.rescale_shape[1])
        self.slist2 = self._create_slist(self.rescale_shape[2])

        self.batch_size = self.raw_img.shape[0]
        self.batch_len = len(self.slist0) + len(self.slist1) + len(self.slist2)

    def __len__(self) -> int:
        """Return batch size * batch length."""
        return self.batch_size * self.batch_len

    def _nn_interpolate(
        self, in_img: torch.Tensor | None = None, **kwargs: dict[str, Any]
    ) -> torch.Tensor | None:
        if not in_img:
            return

        unsqueezed_img = torch.unsqueeze(in_img, 0)
        unsqueezed_img = nn.functional.interpolate(
            unsqueezed_img,
            kwargs.get("rescale_factor"),
            kwargs.get("mode", "Linear"),
            kwargs.get("align_corners", False),
        )
        return torch.squeeze(unsqueezed_img, 0)

    def _create_slist(self, dim_length: int) -> list[range]:
        """Helper to create list of ranges."""
        return [
            range(idx, idx + self.num_slice)
            for idx in range(dim_length - self.num_slice + 1)
        ]

    def get_one_directory(self, axis=0):
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
        for l in slist:
            slice_weight[l] += 1

        slice_data = list()
        for i in ind:
            slice_data.append(self.__getitem__(i))

        return slice_data, slist, slice_weight

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Get dataset blocks for a given index."""
        bind = int(idx / self.batch_len)
        idx = idx % self.batch_len

        if idx < len(self.slist0):
            slice_range = self.slist0[idx]
            raw_img_tmp, bfld_tmp, brainmask_tmp = self._get_slices(
                bind, slice_range, 0
            )
        elif idx < len(self.slist0) + len(self.slist1):
            slice_range = self.slist1[idx - len(self.slist0)]
            raw_img_tmp, bfld_tmp, brainmask_tmp = self._get_slices(
                bind, slice_range, 1
            )
        else:
            slice_range = self.slist2[idx - len(self.slist0) - len(self.slist1)]
            raw_img_tmp, bfld_tmp, brainmask_tmp = self._get_slices(
                bind, slice_range, 2
            )

        raw_img_blk = self._create_block(raw_img_tmp)
        bfld_blk = self._create_block(bfld_tmp, fill_value=1) if bfld_tmp else None
        brainmask_blk = (
            self._create_block(brainmask_tmp, dtype=torch.long)
            if brainmask_tmp
            else None
        )

        if bfld_blk and brainmask_blk:
            return raw_img_blk, bfld_blk, brainmask_blk
        elif bfld_blk:
            return raw_img_blk, bfld_blk
        elif brainmask_blk:
            return raw_img_blk, brainmask_blk
        else:
            return raw_img_blk

    def _get_slices(
        self, bind: int, slice_range: range, axis: int
    ) -> tuple[NDArray, ...]:
        """Helper method to get slices of the data along a specified axis."""
        if axis == 0:
            raw_img_tmp = self.raw_img.data[bind][slice_range, :, :]
            bfld_tmp = (
                self.bfld.data[bind][slice_range, :, :]
                if isinstance(self.bfld, torch.Tensor)
                else None
            )
            brainmask_tmp = (
                self.brainmask.data[bind][slice_range, :, :]
                if isinstance(self.brainmask, torch.Tensor)
                else None
            )
        elif axis == 1:
            raw_img_tmp = self.raw_img.data[bind][:, slice_range, :].permute([1, 0, 2])
            bfld_tmp = (
                self.bfld.data[bind][:, slice_range, :].permute([1, 0, 2])
                if isinstance(self.bfld, torch.Tensor)
                else None
            )
            brainmask_tmp = (
                self.brainmask.data[bind][:, slice_range, :].permute([1, 0, 2])
                if isinstance(self.brainmask, torch.Tensor)
                else None
            )
        elif axis == 2:
            raw_img_tmp = self.raw_img.data[bind][:, :, slice_range].permute([2, 0, 1])
            bfld_tmp = (
                self.bfld.data[bind][:, :, slice_range].permute([2, 0, 1])
                if isinstance(self.bfld, torch.Tensor)
                else None
            )
            brainmask_tmp = (
                self.brainmask.data[bind][:, :, slice_range].permute([2, 0, 1])
                if isinstance(self.brainmask, torch.Tensor)
                else None
            )
        return raw_img_tmp, bfld_tmp, brainmask_tmp

    def _create_block(
        self,
        data_tmp: torch.Tensor,
        extend_dim: int | None = None,
        dtype: torch.dtype = torch.float32,
        fill_value: int = 0,
    ) -> torch.Tensor:
        """Helper to create a block of data with given dimensions and fill values."""
        extend_dim = extend_dim or self.rescale_dim
        block = torch.full(
            [self.num_slice, extend_dim, extend_dim], fill_value, dtype=dtype
        )
        slice_shape = data_tmp.shape
        block[:, : slice_shape[1], : slice_shape[2]] = data_tmp
        return block[:, : slice_shape[1], : slice_shape[2]]


# Unit test?
# if __name__ == "__main__":
#     volume_dataset = VolumeDataset(
#         raw_img_in=None,
#         corrected_img_in="../site-ucdavis/TrainT1w",
#         brainmask_in="../site-ucdavis/TrainMask",
#     )
#     volume_loader = data.DataLoader(dataset=volume_dataset, batch_size=1, shuffle=True)
#     for i, (corrected_img, brainmask) in enumerate(volume_loader):
#         block_dataset = BlockDataset(
#             raw_img=corrected_img,
#             bfld=None,
#             brainmask=brainmask,
#             num_slice=3,
#             rescale_dim=256,
#         )
#         block_loader = data.DataLoader(
#             dataset=block_dataset, batch_size=20, shuffle=True
#         )
#         for j, (corrected_img_blk, brainmask_blk) in enumerate(block_loader):
#             print(brainmask_blk.shape)
