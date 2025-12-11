import torch
from torch import nn
import torchvision


class LocalizationNetwork(torch.torch.nn.Module):
    def __init__(self):
        super().__init__()
        # The localization net uses a downsampled version of the image for performance
        self.input_size = (128, 128)
        self.resize = torchvision.transforms.Resize(
            size=self.input_size, antialias=True
        )
        # Spatial transformer localization-network
        self.localization = torch.nn.Sequential(
            torch.nn.Conv2d(1, 24, kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2, stride=2),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = torch.nn.Sequential(
            torch.nn.Linear(8 * 8 * 64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resized_x = self.resize(x)
        xs = self.localization(resized_x)
        xs = xs.view(-1, 8 * 8 * 64)
        theta_x_y = self.fc_loc(xs)
        theta_x_y = theta_x_y.view(-1, 3)
        theta = theta_x_y[:, 0]  # Rotation angle
        # Construct rotation and scaling matrix
        m11 = torch.cos(theta)
        m12 = -torch.sin(theta)
        m13 = theta_x_y[:, 1]  # offset x
        m21 = torch.sin(theta)
        m22 = torch.cos(theta)
        m23 = theta_x_y[:, 2]  # offset y

        mat = torch.concatenate((m11, m12, m13, m21, m22, m23))
        mat = mat.view(-1, 2, 3)
        grid = torch.nn.functional.affine_grid(mat, x.size(), align_corners=False)
        x = torch.nn.functional.grid_sample(x, grid, align_corners=False)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )  # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,  # value found in tensorflow
            momentum=0.1,  # default pytorch value
            affine=True,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):
    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):
    def __init__(self):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1),
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):
    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):
    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):
    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2),
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):
    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):
    def __init__(self):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2),
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2),
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):
    def __init__(self):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)

        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(
            384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1)
        )
        self.branch1_1b = BasicConv2d(
            384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)
        )

        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(
            384, 448, kernel_size=(3, 1), stride=1, padding=(1, 0)
        )
        self.branch2_2 = BasicConv2d(
            448, 512, kernel_size=(1, 3), stride=1, padding=(0, 1)
        )
        self.branch2_3a = BasicConv2d(
            512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1)
        )
        self.branch2_3b = BasicConv2d(
            512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


DEEPPRINT_INPUT_SIZE = 299


class _InceptionV4_Stem(nn.Module):
    def __init__(self):
        super(_InceptionV4_Stem, self).__init__()
        # Modules
        self.features = nn.Sequential(
            BasicConv2d(1, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
        )

    def forward(self, input):
        assert input.shape[-1] == DEEPPRINT_INPUT_SIZE
        assert input.shape[-2] == DEEPPRINT_INPUT_SIZE
        x = self.features(input)
        return x


class _Branch_TextureEmbedding(nn.Module):
    def __init__(self, texture_embedding_dims: int):
        super(_Branch_TextureEmbedding, self).__init__()
        self._0_block = nn.Sequential(
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(),
        )

        self._1_block = nn.Sequential(
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(),
        )

        self._2_block = nn.Sequential(
            Inception_C(),
            Inception_C(),
            Inception_C(),
        )

        self._3_avg_pool2d = nn.AvgPool2d(
            kernel_size=8
        )  # Might need adjustment if the input size is changed
        self._4_flatten = nn.Flatten()
        self._5_dropout = nn.Dropout(p=0.2)
        self._6_linear = nn.Linear(1536, texture_embedding_dims)

    def forward(self, input):
        x = self._0_block(input)
        x = self._1_block(x)
        x = self._2_block(x)
        x = self._3_avg_pool2d(x)
        x = self._4_flatten(x)
        x = self._5_dropout(x)
        x = self._6_linear(x)
        x = torch.nn.functional.normalize(torch.squeeze(x), dim=1)
        return x


class _Branch_MinutiaStem(nn.Module):
    def __init__(self):
        super().__init__()
        # Modules
        self.features = nn.Sequential(
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
        )

    def forward(self, input):
        return self.features(input)


class _Branch_MinutiaEmbedding(nn.Module):
    def __init__(self, minutia_embedding_dims: int):
        super().__init__()
        # Modules
        self._0_block = nn.Sequential(
            nn.Conv2d(384, 768, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(768, 768, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(768, 896, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(896, 1024, kernel_size=3, stride=2, padding=1),
        )
        self._1_max_pool2d = nn.MaxPool2d(kernel_size=9, stride=1)
        self._2_flatten = nn.Flatten()
        self._3_dropout = nn.Dropout(p=0.2)
        self._4_linear = nn.Linear(1024, minutia_embedding_dims)

    def forward(self, input):
        x = self._0_block(input)
        x = self._1_max_pool2d(x)
        x = self._2_flatten(x)
        x = self._3_dropout(x)
        x = self._4_linear(x)
        x = torch.nn.functional.normalize(x, dim=1)
        return x


class _Branch_MinutiaMap(nn.Module):
    def __init__(self):
        super().__init__()
        # Modules
        self.features = nn.Sequential(
            nn.ConvTranspose2d(384, 128, kernel_size=3, stride=2),
            nn.Conv2d(128, 128, kernel_size=7, stride=1),
            nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2),
            nn.Conv2d(32, 6, kernel_size=3, stride=1),
        )

    def forward(self, input):
        # The network produces maps of size 129x129 but we want maps of size 128x128
        # so we remove the last row / column from each map
        return self.features(input)[:, :, :-1, :-1]


class DeepPrintOutput:
    def __init__(
        self,
        minutia_embeddings: torch.Tensor = None,
        texture_embeddings: torch.Tensor = None,
    ):
        self.minutia_embeddings: torch.Tensor = minutia_embeddings
        self.texture_embeddings: torch.Tensor = texture_embeddings

    @staticmethod
    def training():
        return False


class DeepPrintTrainingOutput(DeepPrintOutput):
    def __init__(
        self,
        minutia_logits: torch.Tensor = None,
        texture_logits: torch.Tensor = None,
        combined_logits: torch.Tensor = None,
        minutia_maps: torch.Tensor = None,
        **kwargs
    ):
        self.minutia_logits: torch.Tensor = minutia_logits
        self.texture_logits: torch.Tensor = texture_logits
        self.combined_logits: torch.Tensor = combined_logits
        self.minutia_maps: torch.Tensor = minutia_maps
        super().__init__(**kwargs)

    @staticmethod
    def training():
        return True


class DeepPrint_Tex(nn.Module):
    """
    Model with only the texture branch.

    In training mode:
        Outputs the texture embedding AND
        A vector of propabilities over all classes (each class being one subject)
    In evaluation mode:
        Outputs the texture embedding
    """

    def __init__(self, num_fingerprints: int, texture_embedding_dims: int):
        super().__init__()
        # Modules
        self.stem = _InceptionV4_Stem()
        self.texture_branch = _Branch_TextureEmbedding(
            texture_embedding_dims=texture_embedding_dims
        )
        self.texture_logits = nn.Sequential(
            nn.Linear(texture_embedding_dims, num_fingerprints), nn.Dropout(p=0.2)
        )

    def forward(self, input) -> torch.Tensor:
        if self.training:
            x = self.stem(input)
            x = self.texture_branch.forward(x)
            logits = self.texture_logits.forward(x)
            return DeepPrintTrainingOutput(texture_logits=logits, texture_embeddings=x)

        with torch.no_grad():
            x = self.stem(input)
            x = self.texture_branch.forward(x)
            return DeepPrintOutput(texture_embeddings=x)


class DeepPrint_Minu(nn.Module):
    """
    Model with only the minutia branch.

    In training mode:
        Outputs the minutia embedding AND
        A vector of propabilities over all classes (each class being one subject) AND
        Predicted minutia maps
    In evaluation mode:
        Outputs the minutia embeddings
    """

    def __init__(self, num_fingerprints: int, minutia_embedding_dims: int):
        super().__init__()
        # Modules
        self.stem = _InceptionV4_Stem()
        self.minutia_stem = _Branch_MinutiaStem()
        self.minutia_map = _Branch_MinutiaMap()
        self.minutia_embedding = _Branch_MinutiaEmbedding(minutia_embedding_dims)
        self.minutia_logits = nn.Sequential(
            nn.Linear(minutia_embedding_dims, num_fingerprints), nn.Dropout(p=0.2)
        )

    def forward(self, input: torch.Tensor) -> DeepPrintOutput:
        if self.training:
            x = self.stem(input)

            x_minutia = self.minutia_stem.forward(x)
            x_minutia_emb = self.minutia_embedding.forward(x_minutia)
            x_minutia_logits = self.minutia_logits(x_minutia_emb)
            x_minutia_map = self.minutia_map(x_minutia)

            return DeepPrintTrainingOutput(
                minutia_logits=x_minutia_logits,
                minutia_maps=x_minutia_map,
                minutia_embeddings=x_minutia_emb,
            )

        with torch.no_grad():
            x = self.stem(input)
            x_minutia = self.minutia_stem.forward(x)

            x_minutia_emb = self.minutia_embedding.forward(x_minutia)
            return DeepPrintOutput(minutia_embeddings=x_minutia_emb)


class DeepPrint_TexMinu(nn.Module):
    """
    Model with texture and minutia branch

    In training mode:
        Outputs the texture embedding AND
        A vector of propabilities over all classes predicted from the texture embedding
    In evaluation mode:
        Outputs the texture embedding

    """

    def __init__(
        self, num_fingerprints, texture_embedding_dims: int, minutia_embedding_dims: int
    ):
        super().__init__()
        # Modules
        self.stem = _InceptionV4_Stem()
        self.texture_branch = _Branch_TextureEmbedding(texture_embedding_dims)
        self.texture_logits = nn.Sequential(
            nn.Linear(texture_embedding_dims, num_fingerprints), nn.Dropout(p=0.2)
        )
        self.minutia_stem = _Branch_MinutiaStem()
        self.minutia_map = _Branch_MinutiaMap()
        self.minutia_embedding = _Branch_MinutiaEmbedding(minutia_embedding_dims)
        self.minutia_logits = nn.Sequential(
            nn.Linear(minutia_embedding_dims, num_fingerprints), nn.Dropout(p=0.2)
        )

    def forward(self, input: torch.Tensor) -> DeepPrintOutput:
        if self.training:
            x = self.stem(input)

            x_texture_emb = self.texture_branch.forward(x)
            x_texture_logits = self.texture_logits(x_texture_emb)

            x_minutia = self.minutia_stem.forward(x)
            x_minutia_emb = self.minutia_embedding.forward(x_minutia)
            x_minutia_logits = self.minutia_logits(x_minutia_emb)
            x_minutia_map = self.minutia_map(x_minutia)

            return DeepPrintTrainingOutput(
                minutia_logits=x_minutia_logits,
                texture_logits=x_texture_logits,
                minutia_maps=x_minutia_map,
                minutia_embeddings=x_minutia_emb,
                texture_embeddings=x_texture_emb,
            )

        with torch.no_grad():
            x = self.stem(input)
            x_texture_emb = self.texture_branch.forward(x)
            x_minutia = self.minutia_stem.forward(x)

            x_minutia_emb = self.minutia_embedding.forward(x_minutia)
            return DeepPrintOutput(x_minutia_emb, x_texture_emb)


class DeepPrint_LocTex(nn.Module):
    """
    Model with texture branch and localization network.

    In training mode:
        Outputs the texture embedding AND
        A vector of propabilities over all classes predicted from the texture embedding
    In evaluation mode:
        Outputs the texture embedding

    """

    def __init__(self, num_fingerprints, texture_embedding_dims: int):
        super().__init__()
        # Special attributs
        self.input_space = None

        # Modules
        self.localization = LocalizationNetwork()
        self.embeddings = DeepPrint_Tex(
            num_fingerprints=num_fingerprints,
            texture_embedding_dims=texture_embedding_dims,
        )

    def forward(self, input: torch.Tensor) -> DeepPrintOutput:
        if self.training:
            x = self.localization(input)
            self.embeddings.train()
            return self.embeddings(x)

        with torch.no_grad():
            x = self.localization(input)
            self.embeddings.eval()
            return self.embeddings(x)


class DeepPrint_LocMinu(nn.Module):
    """
    Model with minutia branch and localization network.

    In training mode:
        Outputs the minutia embedding AND
        A vector of propabilities over all classes predicted from the minutia embedding AND
        The generated minutia maps
    In evaluation mode:
        Outputs the minutia embedding

    """

    def __init__(self, num_fingerprints, minutia_embedding_dims: int):
        super().__init__()
        # Special attributs
        self.input_space = None

        # Modules
        self.localization = LocalizationNetwork()
        self.embeddings = DeepPrint_Minu(
            num_fingerprints=num_fingerprints,
            minutia_embedding_dims=minutia_embedding_dims,
        )

    def forward(self, input: torch.Tensor) -> DeepPrintOutput:
        if self.training:
            x = self.localization(input)
            self.embeddings.train()
            return self.embeddings(x)

        with torch.no_grad():
            x = self.localization(input)
            self.embeddings.eval()
            return self.embeddings(x)


class DeepPrint_LocTexMinu(nn.Module):
    """
    Model with texture and minutia branch and localization network.

    In training mode:
        Outputs the texture embedding AND
        A vector of propabilities over all classes predicted from the texture embedding
    In evaluation mode:
        Outputs the texture embedding

    """

    def __init__(
        self, num_fingerprints, texture_embedding_dims: int, minutia_embedding_dims: int
    ):
        super().__init__()
        # Special attributs
        self.input_space = None

        # Modules
        self.localization = LocalizationNetwork()
        self.embeddings = DeepPrint_TexMinu(
            num_fingerprints=num_fingerprints,
            texture_embedding_dims=texture_embedding_dims,
            minutia_embedding_dims=minutia_embedding_dims,
        )

    def forward(self, input: torch.Tensor) -> DeepPrintOutput:
        if self.training:
            x = self.localization(input)
            self.embeddings.train()
            return self.embeddings(x)

        with torch.no_grad():
            x = self.localization(input)
            self.embeddings.eval()
            return self.embeddings(x)


def main():

    models = [
    DeepPrint_LocTexMinu(400, 96, 96)
    ]

    for model in models:
        p = 0
        for param in model.parameters():
            p += param.numel()
        print(p)

if __name__ == "__main__":
    main()