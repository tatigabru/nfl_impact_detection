import os
import imageio
import math
from scipy.ndimage import zoom
import glob
from volumentations import *


class ChannelShuffle(DualTransform):
    def __init__(self, order=(2, 1, 0), always_apply=False, p=0.3):
        super().__init__(always_apply, p)
        self.order = order

    def apply(self, img):
        return img[:, :, :, self.order]


aug_list_original = [
        Rotate((0, 0), (0, 0), (10, 10), p=0.3),
        ElasticTransform((0, 0.1), interpolation=1, p=0.1),
        RandomCropFromBorders(crop_value=0.05, p=0.4),
        RandomDropPlane(plane_drop_prob=0.1, axes=(0, 1, 2), p=0.5),
        #Resize(patch_size, interpolation=1, always_apply=True,p=1.0),
        GridDropout(unit_size_min=2, unit_size_max=6,holes_number_x=5, holes_number_y=5, holes_number_z=5,random_offset=True, p=0.1),
        #Flip(0, p=0.5),
        Flip(1, p=0.5),
        #Flip(2, p=0.5),
        #RandomRotate90((1, 2), p=0.5),
        GaussianNoise(var_limit=(0, 5), p=0.2),
        RandomGamma(gamma_limit=(0.5, 1.5), p=0.2)
    ]

aug_list = [
        Rotate((0, 0), (0, 0), (10, 10), p=1),
        ElasticTransform((0, 0.1), interpolation=1, p=1),
        RandomCropFromBorders(crop_value=0.1, p=1),
        #RandomDropPlane(plane_drop_prob=0.1, axes=(0, 1, 2),p=0.5),
        Resize((80, 80, 16), interpolation=1, always_apply=True,p=1.0),
        GridDropout(unit_size_min=2, unit_size_max=50,holes_number_x=5, holes_number_y=5, holes_number_z=5,random_offset=True, p=1),
        Flip(0, p=1),
        Flip(1, p=1),
        Flip(2, p=1),
        GaussianNoise(var_limit=(0, 5), p=1),
        RandomGamma(gamma_limit=(0.5, 1.5), p=1),
        ChannelShuffle(order=(2, 1, 0), p=1)
    ]


def get_augmentation():
    return Compose([
        Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
        RandomCropFromBorders(crop_value=0.1, p=0.5),
        ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        Flip(2, p=0.5),
        RandomRotate90((1, 2), p=0.5),
        GaussianNoise(var_limit=(0, 5), p=0.2),
        RandomGamma(gamma_limit=(0.5, 1.5), p=0.2),
    ], p=1.0)

aug = get_augmentation()

# with mask
img = np.zeros((80, 80, 16))
lbl = np.ones((80, 80, 16))
data = {'image': img, 'mask': lbl}
aug_data = aug(**data)
img, lbl = aug_data['image'], aug_data['mask']

# without mask
data = {'image': img}
aug_data = aug(**data)
img = aug_data['image']


def read_tensor():
    tensor_fp = "/home/anastasiya/Kaggle/SSD_link/tensors/train/uint/small/57583_000082_Endzone.mp4/680.npy"
    tensor = np.load(tensor_fp)
    return tensor


def save_mini_clip(tensor: np.array, data_dir, unique_id, has_overlap, label):
    cache_dir = os.path.join(data_dir, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    has_impact_sfx = '_impact' if label == 1 else ''
    has_overlap_sfx = '_overlap' if has_overlap == 1 else ''
    base_path = os.path.join(cache_dir, str(unique_id) + has_overlap_sfx + has_impact_sfx)
    video_path = base_path + '.gif'
    mosaic_path = base_path + '.png'

    mosaic = make_mosaic_from_tensor(tensor)
    #mosaic = denormalize(mosaic, DATASET_MEAN, DATASET_STD).astype(np.uint8)
    print(mosaic_path)
    imageio.imwrite(mosaic_path, mosaic)

    n, width, height, n_channels = tensor.shape
    fps = 10
    upscale = 4
    images = []
    for i in range(n):
        #image = denormalize(tensor[i], DATASET_MEAN, DATASET_STD).astype(np.uint8)
        image = tensor[i]
        image = zoom(image, (upscale, upscale, 1))
        images.append(image)
    imageio.mimsave(video_path, images, fps=fps)
    return video_path, mosaic_path


def make_mosaic_from_tensor(tensor: np.ndarray, border=1):
    n = len(tensor)
    images = [tensor[i] for i in range(n)]
    border_array = [1, 1, 1, 1] * border
    images_ = [cv2.copyMakeBorder(image, *border_array, borderType=cv2.BORDER_CONSTANT) for image in images]
    s0, s1, s2 = images[0].shape
    image_padding = np.zeros((s0 + 2 * border,
                              s1 + 2 * border, s2))

    n0 = math.ceil(math.sqrt(n))
    n1 = math.ceil(len(tensor) / n0)
    padding_images = [image_padding for _ in range(n0 * n1 - n)]
    all_images = images_ + padding_images
    mosaic = np.concatenate([np.concatenate([all_images[n0 * k + i] for i in range(n0)], axis=1) for k in range(n1)], axis=0)
    return mosaic


def denormalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    img = img.astype(np.float32)
    img *= std
    img += mean
    return img


def test_augmentations(data_dir):

    cache_dir = os.path.join(data_dir, 'cache')
    files = glob.glob(os.path.join(os.path.join(data_dir, 'cache'), '*'))
    for f in files:
        os.remove(f)

    tensor = np.zeros((80, 80, 16, 3), dtype=np.uint8)
    tensor[8: 72, 8: 72, :, 0] = 0
    tensor[8: 72, 8: 72, :, 0] = 0
    tensor[8: 72, 8: 72, :, 0] = 255
    tensor = read_tensor()
    tensor = np.transpose(tensor, (2, 3, 1, 0))
    print(np.min(tensor))
    print(np.max(tensor))
    for i, aug in enumerate(aug_list):
        print(str(aug))
        print(tensor.shape)
        data = {'image': tensor}
        compose_aug = Compose([aug], p=1)
        aug_data = compose_aug(**data)
        aug_tensor = aug_data['image']
        print(np.min(aug_tensor))
        print(np.max(aug_tensor))
        print(aug_tensor.shape)
        save_mini_clip(np.transpose(aug_tensor, (2, 0, 1, 3)), data_dir, str(aug) + str(i), 0, 0)


if __name__ == '__main__':
    data_dir = os.environ["KAGGLE_2020_NFL"]
    test_augmentations(data_dir)
