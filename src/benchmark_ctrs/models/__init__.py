from lightning.pytorch.utilities import LightningEnum


class Architectures(LightningEnum):
    LeNet = "lenet"
    Resnet50 = "resnet50"
    Resnet110 = "resnet110"
