"""Module associated with creating torch models."""

import torch
import torch.nn as nn


def Conv3dBlock(
    dim_in: int,
    dim_out: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = True,
    use_batch_norm: bool = False,
) -> nn.Module:
    """Create a 3D convolution block."""
    nn_modules = []
    nn_modules.append(
        nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
    )

    if use_batch_norm:
        nn_modules.append(nn.BatchNorm3d(dim_out))

    nn_modules.append(nn.LeakyReLU(0.1))
    nn_modules.append(
        nn.Conv3d(
            dim_out,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    )
    if use_batch_norm:
        nn_modules.append(nn.BatchNorm3d(dim_out))
    nn_modules.append(nn.LeakyReLU(0.1))

    return nn.Sequential(*nn_modules)


def UpConv3dBlock(
    dim_in: int,
    dim_out: int,
    kernel_size: int = 4,
    stride: int = 2,
    padding: int = 1,
    bias: bool = False,
) -> nn.Module:
    """Create a 3D up-convolution block."""
    return nn.Sequential(
        nn.ConvTranspose3d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        nn.LeakyReLU(0.1),
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
    """Create a 2D convolution block."""
    nn_modules = []
    nn_modules.append(
        nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    )
    if use_batch_norm:
        nn_modules.append(nn.BatchNorm2d(dim_out))
    nn_modules.append(nn.LeakyReLU(0.1))
    nn_modules.append(
        nn.Conv2d(
            dim_out,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    )
    if use_batch_norm:
        nn_modules.append(nn.BatchNorm2d(dim_out))
    nn_modules.append(nn.LeakyReLU(0.1))

    return nn.Sequential(*nn_modules)


def UpConv2dBlock(
    dim_in: int,
    dim_out: int,
    kernel_size: int = 4,
    stride: int = 2,
    padding: int = 1,
    bias: bool = True,
) -> nn.Module:
    """Create a 2D up-convolution block."""
    return nn.Sequential(
        nn.ConvTranspose2d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        nn.LeakyReLU(0.1),
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

        self.layers = dict()
        self.num_conv_block = num_conv_block
        # Convolution layers
        for num in range(1, self.num_conv_block + 1):
            if num == 1:
                setattr(
                    self,
                    "conv%d" % (num),
                    Conv3dBlock(dim_in, kernel_root, use_batch_norm=use_batch_norm),
                )
            else:
                setattr(
                    self,
                    "conv%d" % (num),
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
                "upconv%dto%d" % (idx + 1, idx),
                UpConv3dBlock(kernel_root * (2**idx), kernel_root * (2 ** (idx - 1))),
            )
            setattr(
                self,
                "conv%dm" % (idx),
                Conv3dBlock(kernel_root * (2**idx), kernel_root * (2 ** (idx - 1))),
            )
        setattr(self, "max_pool", nn.MaxPool3d(2))
        setattr(self, "out_layer", nn.Conv3d(kernel_root, 2, 3, 1, 1))

        # Weight Initialization
        for model in self.modules():
            if isinstance(model, nn.Conv3d) or isinstance(model, nn.ConvTranspose3d):
                model.weight.data.normal_(0, 0.02)
                if model.bias is not None:
                    model.bias.data.zero_()
            elif isinstance(model, nn.BatchNorm3d):
                model.weight.data.normal_(1.0, 0.02)

    def forward(self, layer: nn.Module) -> nn.Module:
        """Method for moving forward."""
        num_conv_block = self.num_conv_block
        conv_out = dict()
        for num in range(1, num_conv_block + 1):
            if num == 1:
                conv_out["conv%d" % (num)] = getattr(self, "conv%d" % (num + 1))(layer)
            else:
                conv_out["conv%d" % (num + 1)] = getattr(self, "conv%d" % (num + 1))(
                    self.max_pool(conv_out["conv%d" % num])
                )

        for num in range(num_conv_block - 1):
            idx = num_conv_block - 1 - num
            tmp = torch.cat(
                (
                    getattr(self, "upconv%dto%d" % (idx + 1, idx))(
                        conv_out["conv%d" % (idx + 1)]
                    ),
                    conv_out["conv%d" % (idx)],
                ),
                1,
            )
            out = getattr(self, "conv%dm" % (idx))(tmp)

        out = self.out_layer(out)
        if not self.training:
            softmax_layer = nn.Softmax(dim=1)
            out = softmax_layer(out)
        return out


class UNet2d(nn.Module):
    """Representation of a 2D UNet."""

    def __init__(
        self,
        dim_in: int = 6,
        num_conv_block: int = 3,
        kernel_root: int = 4,
        use_batch_norm: bool = True,
    ) -> None:
        super(UNet2d, self).__init__()

        self.layers = dict()
        self.num_conv_block = num_conv_block

        # Convolution Layers
        for num in range(1, num_conv_block + 1):
            if num == 1:
                setattr(
                    self,
                    "conv%d" % (num),
                    Conv2dBlock(dim_in, kernel_root, use_batch_norm=use_batch_norm),
                )
            else:
                setattr(
                    self,
                    "conv%d" % (num),
                    Conv2dBlock(
                        kernel_root * (2 ** (num - 1)),
                        kernel_root * (2**num),
                        use_batch_norm=use_batch_norm,
                    ),
                )

        # Up-convolution Layers
        for num in range(num_conv_block - 1):
            idx = num_conv_block - 1 - num
            setattr(
                self,
                "upconv%dto%d" % (idx + 1, idx),
                UpConv2dBlock(kernel_root * (2**idx), kernel_root * (2 ** (idx - 1))),
            )
            setattr(
                self,
                "conv%dm" % (idx),
                Conv2dBlock(kernel_root * (2**idx), kernel_root * (2 ** (idx - 1))),
            )
        setattr(self, "max_pool", nn.MaxPool2d(2))
        setattr(self, "out_layer", nn.Conv2d(kernel_root, 2, 3, 1, 1))

        # Weight Initialization
        self.apply(self.weights_init)

    def weights_init(self, model: nn.Module) -> None:
        """Initialize model weights."""
        if isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d):
            model.weight.data.normal_(0, 0.02)
            if model.bias is not None:
                model.bias.data.zero_()
        elif isinstance(model, nn.BatchNorm2d):
            model.weight.data.normal_(1.0, 0.02)

    def forward(self, layer: nn.Module) -> nn.Module:
        """Method for moving forward in model."""
        num_conv_block = self.num_conv_block
        conv_out = dict()
        for num in range(1, num_conv_block + 1):
            if num == 1:
                conv_out["conv%d" % (num)] = getattr(self, "conv%d" % (num))(layer)
            else:
                conv_out["conv%d" % (num)] = getattr(self, "conv%d" % (num))(
                    self.max_pool(conv_out["conv%d" % num])
                )

        for num in range(num_conv_block - 1):
            idx = num_conv_block - 1 - num
            if num == 0:
                tmp = torch.cat(
                    (
                        getattr(self, "upconv%dto%d" % (idx + 1, idx))(
                            conv_out["conv%d" % (idx + 1)]
                        ),
                        conv_out["conv%d" % (idx)],
                    ),
                    1,
                )
            else:
                tmp = torch.cat(
                    (
                        getattr(self, "upconv%dto%d" % (idx + 1, idx))(
                            conv_out["conv%d" % (idx + 1)]
                        ),
                        conv_out["conv%d" % (idx)],
                    ),
                    1,
                )

            out = getattr(self, "conv%dm" % (idx))(tmp)

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

        for i in range(num_slice):
            setattr(
                self,
                "slice%d" % (i + 1),
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

    def forward(self, layer: nn.Module) -> nn.Module:
        """Method for moving forward in model."""
        for idx in range(1, self.num_slice + 1):
            pho = getattr(self, "slice%d" % (idx + 1))(layer)
            if idx == 0:
                out = pho
            else:
                out = torch.cat((out, pho), 1)
        return out

    def freeze(self, freeze: bool = False) -> None:
        """Method for (un)freezing parameters.

        NOTE: This is not used anywhere (model isn't included).
        """
        for param in model.parameters():  # noqa: F821
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

        for i in range(num_slice):
            setattr(
                self,
                "slice%d" % (i + 1),
                UNet2d(
                    dim_in=num_slice,
                    num_conv_block=num_conv_block,
                    kernel_root=kernel_root,
                    use_batch_norm=use_batch_norm,
                ),
            )

        self.num_slice = num_slice

    def forward(self, layer: nn.Module) -> nn.Module:
        """Method for moving forward in model."""
        for idx in range(1, self.num_slice + 1):
            pho = torch.unsqueeze(getattr(self, "slice%d" % (idx))(layer), 2)
            if idx == 1:
                out = pho
            else:
                out = torch.cat((out, pho), 2)
        return out

    def freeze(self, freeze: bool = False) -> None:
        """Method for (un)freezing model.

        NOTE: Currently not used as model isn't provided.
        """
        for param in model.parameters():  # noqa: F821
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

    def forward(self, layer: nn.Module, model: str = "forward_full") -> nn.Module:
        """Method for moving forward in model."""
        if model == "forward_bc_part":
            b_field = self.BcUNet(layer)
            out = b_field
        elif model == "forward_ss_part":
            b_msk = self.SsUNet(layer)
            out = b_msk
        elif model == "forward_full":
            b_field = self.BcUNet(layer)
            layer = layer * b_field
            out = self.SsUNet(layer)
        else:
            raise NotImplementedError("Model not recognized.")
        return out


# Unit test?
# if __name__=='__main__':
#     model=UNet2d(dim_in=3)

#     x=Variable(torch.rand(2, 3, 256, 256))

#     model.cuda()
#     x=x.cuda()

#     h_x=model(x)
#     print(h_x.shape)
# model.BcUNet.unfreeze()
# for child in model.BcUNet.children():
#    for param in child.parameters():
#        print(param.requires_grad)
# model.cuda()
# cudnn.benchmark=True
# x=x.cuda()
# h_x=model(x)
