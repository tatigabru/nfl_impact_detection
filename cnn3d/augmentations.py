from volumentations import *
import albumentations as A
from scipy.ndimage import zoom



DATASET_MEAN = (0.485, 0.456, 0.406)
DATASET_STD = (0.229, 0.224, 0.225)


# random resized crop
# resize


def pad_tensor():
    return Compose([PadIfNeeded((64, 64, 16))], p=1)


class ChannelShuffle(DualTransform):
    def __init__(self, order=(2, 1, 0), always_apply=False, p=0.3):
        super().__init__(always_apply, p)
        self.order = order

    def apply(self, img):
        return img[:, :, :, self.order]


def get_augmentations_v1():
        aug_list = [
            Rotate((0, 0), (0, 0), (10, 10), p=0.3),
            #ElasticTransform((0, 0.1), interpolation=1, p=0.1),
            #RandomCropFromBorders(crop_value=0.05, p=0.4),
            #RandomDropPlane(plane_drop_prob=0.1, axes=(0, 1, 2), p=0.5),
            #Resize((80, 80, 16), interpolation=1, always_apply=True, p=1.0),
            GridDropout(unit_size_min=2, unit_size_max=6, holes_number_x=5, holes_number_y=5, holes_number_z=5,
                       random_offset=True, p=0.1),
            # Flip(0, p=0.5),
            Flip(1, p=0.5),
            # Flip(2, p=0.5),
            # RandomRotate90((1, 2), p=0.5),
            GaussianNoise(var_limit=(0, 5), p=0.2),
            RandomGamma(gamma_limit=(0.5, 1.5), p=0.2),
            ChannelShuffle(order=(2, 1, 0), p=0.3),
            RandomCrop((76, 76, 16))
        ]
        crop = Compose([CenterCrop((64, 64, 16), p=1)], p=1)
        norm = A.Compose([A.Normalize(p=1)], p=1)
        return Compose(aug_list, p=1), crop, norm

