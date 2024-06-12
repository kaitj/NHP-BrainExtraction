"""Module associated with creating torch models."""

from __future__ import annotations

import torch
import torch.nn as nn


def create_conv_block(
    dim_in: int,
    dim_out: int,
    kernel_size: int,
    stride: int,
    padding: int,
    bias: bool,
    use_batch_norm: bool,
    conv_type: str,
) -> nn.Module:
    """Helper function to create convolution block."""
    if conv_type.lower() == "2d":
        conv_class = nn.Conv2d
        batch_norm_class = nn.BatchNorm2d
    elif conv_type.lower() == "3d":
        conv_class = nn.Conv3d
        batch_norm_class = nn.BatchNorm3d
    else:
        raise ValueError(f"Invalid convolution type: {conv_type}.")

    layers: list[nn.Module] = [
        conv_class(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        nn.LeakyReLU(0.1),
        conv_class(
            dim_out,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        nn.LeakyReLU(0.1),
    ]

    if use_batch_norm:
        layers.insert(1, batch_norm_class(dim_out))
        layers.insert(4, batch_norm_class(dim_out))

    return nn.Sequential(*layers)


def create_upconv_block(
    dim_in: int,
    dim_out: int,
    kernel_size: int,
    stride: int,
    padding: int,
    bias: bool,
    conv_type: str = "2d",
) -> nn.Module:
    """Helper function to create up-convolution block."""
    if conv_type.lower() == "2d":
        conv_transpose_class = nn.ConvTranspose2d
    elif conv_type.lower() == "3d":
        conv_transpose_class = nn.ConvTranspose3d
    else:
        raise ValueError(f"Invalid convolution type: {conv_type}.")

    return nn.Sequential(
        conv_transpose_class(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        nn.LeakyReLU(0.1),
    )


def Conv3dBlock(
    dim_in: int,
    dim_out: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = True,
    use_batch_norm: bool = False,
) -> nn.Module:
    """Create 3D convolution block."""
    return create_conv_block(
        dim_in=dim_in,
        dim_out=dim_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        use_batch_norm=use_batch_norm,
        conv_type="3d",
    )


def UpConv3dBlock(
    dim_in: int,
    dim_out: int,
    kernel_size: int = 4,
    stride: int = 2,
    padding: int = 1,
    bias: bool = False,
) -> nn.Module:
    """Create 3D up-convolution block."""
    return create_upconv_block(
        dim_in=dim_in,
        dim_out=dim_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        conv_type="3d",
    )


def Conv2dBlock(
    dim_in: int,
    dim_out: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = True,
    use_batch_norm: bool = True,
) -> nn.Module:
    """Create 2D convolution block."""
    return create_conv_block(
        dim_in=dim_in,
        dim_out=dim_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        use_batch_norm=use_batch_norm,
        conv_type="2d",
    )


def UpConv2dBlock(
    dim_in: int,
    dim_out: int,
    kernel_size: int = 4,
    stride: int = 2,
    padding: int = 1,
    bias: bool = True,
) -> nn.Module:
    """Create 2D up-convolution block."""
    return create_upconv_block(
        dim_in=dim_in,
        dim_out=dim_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        conv_type="2d",
    )


class UNet3d(nn.Module):
    """Representation of a 3D UNet."""

    def __init__(
        self,
        dim_in: int = 1,
        num_conv_block: int = 2,
        kernel_root: int = 8,
        use_batch_norm: bool = False,
    ) -> None:
        super(UNet3d, self).__init__()
        self.num_conv_block = num_conv_block
        self.layers: dict[str, nn.Module] = {}

        # Convolution layers
        for num in range(num_conv_block):
            if num == 0:
                setattr(
                    self,
                    f"conv{num+1}",
                    Conv3dBlock(dim_in, kernel_root, use_batch_norm=use_batch_norm),
                )
            else:
                setattr(
                    self,
                    f"conv{num+1}",
                    Conv3dBlock(
                        kernel_root * (2 ** (num - 1)),
                        kernel_root * (2**num),
                        use_batch_norm=use_batch_norm,
                    ),
                )

        # Up-convolution layers
        for num in range(num_conv_block - 1):
            idx = num_conv_block - 1 - num
            setattr(
                self,
                f"upconv{idx+1}to{idx}",
                UpConv3dBlock(kernel_root * (2**idx), kernel_root * (2 ** (idx - 1))),
            )
            setattr(
                self,
                f"conv{idx}m",
                Conv3dBlock(kernel_root * (2**idx), kernel_root * (2 ** (idx - 1))),
            )

        setattr(self, "max_pool", nn.MaxPool3d(2))
        setattr(self, "out_layer", nn.Conv3d(kernel_root, 2, 3, 1, 1))

        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module) -> None:
        """Initialize module weights."""
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            module.weight.data.normal_(0, 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm3d):
            module.weight.data.norm_(1.0, 0.02)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Method for moving model forward."""
        conv_out: dict[str, torch.Tensor] = {}

        # Convolution in UNet
        for num in range(self.num_conv_block):
            if num == 0:
                conv_out[f"conv{num+1}"] = getattr(self, f"conv{num+1}")(tensor)
            else:
                conv_out[f"conv{num+1}"] = getattr(self, f"conv{num+1}")(
                    self.max_pool(conv_out[f"conv{num}"])
                )

        # Up-convolution in UNet
        for num in range(self.num_conv_block - 1):
            idx = self.num_conv_block - 1 - num
            tmp = torch.cat(
                (
                    getattr(self, f"upconv{idx+1}to{idx}")(conv_out[f"conv{idx+1}"]),
                    conv_out[f"conv{idx}"],
                ),
                1,
            )
            out = getattr(self, f"conv{idx}m")(tmp)

        out = self.out_layer(out)
        if not self.training:
            softmax_layer = nn.Softmax(1)
            out = softmax_layer(out)
        return out


class UNet2d(nn.Module):
    """Representation of 2D UNet."""

    def __init__(
        self,
        dim_in: int = 6,
        num_conv_block: int = 3,
        kernel_root: int = 4,
        use_batch_norm: bool = True,
    ) -> None:
        super(UNet2d, self).__init__()
        self.num_conv_block = num_conv_block
        self.layers: dict[str, nn.Module] = {}

        # Convolution blocks
        for num in range(num_conv_block):
            if num == 0:
                setattr(
                    self,
                    f"conv{num+1}",
                    Conv2dBlock(dim_in, kernel_root, use_batch_norm=use_batch_norm),
                )
            else:
                setattr(
                    self,
                    f"conv{num+1}",
                    Conv2dBlock(
                        kernel_root * (2 ** (num - 1)),
                        kernel_root * (2**num),
                        use_batch_norm=use_batch_norm,
                    ),
                )

        # Up-convolution blocks
        for num in range(num_conv_block - 1):
            idx = num_conv_block - 1 - num
            setattr(
                self,
                f"upconv{idx+1}to{idx}",
                UpConv2dBlock(kernel_root * (2**idx), kernel_root * (2 ** (idx - 1))),
            )
            setattr(
                self,
                f"conv{idx}m",
                Conv2dBlock(kernel_root * (2**idx), kernel_root * (2 ** (idx - 1))),
            )

        setattr(self, "max_pool", nn.MaxPool2d(2))
        setattr(self, "out_layer", nn.Conv2d(kernel_root, 2, 3, 1, 1))

        self.apply(self.weights_init)

    def weights_init(self, module: nn.Module) -> None:
        """Initialize module weights."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(0, 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.normal_(1.0, 0.02)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Method for moving model forward."""
        conv_out: dict[str, torch.Tensor] = {}

        # Convolution layers
        for num in range(self.num_conv_block):
            if num == 0:
                conv_out[f"conv{num+1}"] = getattr(self, f"conv{num+1}")(tensor)
            else:
                conv_out[f"conv{num+1}"] = getattr(self, f"conv{num+1}")(
                    self.max_pool(conv_out[f"conv{num}"])
                )

        # Up-convolution layers
        for num in range(self.num_conv_block - 1):
            idx = self.num_conv_block - 1 - num
            if num == 0:
                tmp = torch.cat(
                    (
                        getattr(self, f"upconv{idx+1}to{idx}")(
                            conv_out[f"conv{idx+1}"]
                        ),
                        conv_out[f"conv{idx}"],
                    ),
                    1,
                )
            else:
                tmp = torch.cat(
                    (
                        getattr(self, f"upconv{idx+1}to{idx}")(out),
                        conv_out[f"conv{idx}"],
                    ),
                    1,
                )

            out = getattr(self, f"conv{idx}m")(tmp)

        out = self.out_layer(out)
        return out


class MultiSliceBcUNet(nn.Module):
    """Representation of a multislice bidirectional C-LSTM (BC) UNet."""

    def __init__(
        self,
        num_slice: int = 6,
        num_conv_block: int = 4,
        kernel_root: int = 16,
        use_batch_norm: bool = True,
    ) -> None:
        super(MultiSliceBcUNet, self).__init__()

        for idx in range(num_slice):
            setattr(
                self,
                f"slice{idx+1}",
                nn.Sequential(
                    UNet2d(
                        dim_in=num_slice,
                        num_conv_block=num_conv_block,
                        kernel_root=kernel_root,
                        use_batch_norm=use_batch_norm,
                    ),
                    nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                ),
            )
        self.num_slice = num_slice

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Method for moving forward in model."""
        for idx in range(self.num_slice):
            pho = getattr(self, f"slice{idx+1}")(tensor)
            if idx == 0:
                out = pho
            else:
                out = torch.cat((out, pho), 1)
        return out

    def freeze(self, freeze: bool = False) -> None:
        """Method for (un)freezing parameters.

        NOTE: This is not used anywehre (model isn't included).
        """
        for param in model.parameters():
            param.requires_grad = freeze


class MultiSliceSsUNet(nn.Module):
    """Representation of a multi-slice, slice shift UNet."""

    def __init__(
        self,
        num_slice: int = 6,
        num_conv_block: int = 5,
        kernel_root: int = 16,
        use_batch_norm: bool = True,
    ) -> None:
        super(MultiSliceSsUNet, self).__init__()

        for idx in range(num_slice):
            setattr(
                self,
                f"slice{idx+1}",
                UNet2d(
                    dim_in=num_slice,
                    num_conv_block=num_conv_block,
                    kernel_root=kernel_root,
                    use_batch_norm=use_batch_norm,
                ),
            )
        self.num_slice = num_slice

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Method for moving forward in model."""
        for idx in range(self.num_slice):
            pho = torch.unsqueeze(getattr(self, f"slice{idx+1}")(tensor), 2)
            if idx == 0:
                out = pho
            else:
                out = torch.cat((out, pho), 2)
        return out

    def freeze(self, freeze: bool = False) -> None:
        """Method for (un)freezing model.

        NOTE: Currently not used as model isn't provided.
        """
        for param in model.parameters():
            param.requires_grad = freeze


class MultiSliceModel(nn.Module):
    """Representation of a multi slice model."""

    def __init__(
        self,
        num_slice: int = 6,
        bc_num_conv_block: int = 3,
        bc_kernel_root: int = 8,
        ss_num_conv_block: int = 4,
        ss_kernel_root: int = 8,
        use_batch_norm: bool = True,
    ) -> None:
        super(MultiSliceModel, self).__init__()

        self.BcUNet = MultiSliceBcUNet(
            num_slice=num_slice,
            num_conv_block=bc_num_conv_block,
            kernel_root=bc_kernel_root,
            use_batch_norm=use_batch_norm,
        )
        self.SsUNet = MultiSliceSsUNet(
            num_slice=num_slice,
            num_conv_block=ss_num_conv_block,
            kernel_root=ss_kernel_root,
            use_batch_norm=use_batch_norm,
        )

    def forward(
        self, tensor: torch.Tensor, model: str = "forward_full"
    ) -> torch.Tensor:
        """Method for moving forward in model."""
        if model == "forward_bc_part":
            b_field = self.BcUNet(tensor)
            out = b_field
        elif model == "forward_ss_part":
            b_msk = self.SsUNet(tensor)
            out = b_msk
        elif model == "forward_full":
            b_field = self.BcUNet(tensor)
            tensor = tensor * b_field
            out = self.SsUNet(tensor)
        return out


# if __name__ == "__main__":
#     model = UNet2d(dim_in=3)

#     x = Variable(torch.rand(2, 3, 256, 256))

#     model.cuda()
#     x = x.cuda()

#     h_x = model(x)
#     print(h_x.shape)
# model.BcUNet.unfreeze()
# for child in model.BcUNet.children():
#    for param in child.parameters():
#        print(param.requires_grad)
# model.cuda()
# cudnn.benchmark=True
# x=x.cuda()
# h_x=model(x)
