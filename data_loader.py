from PIL import Image
from torch.utils.data import Dataset, DataLoader

from augmentations import ha_augment_sample, resize_sample, spatial_augment_sample
from utils import to_tensor_sample

# 对图像进行一系列处理
def image_transforms(shape, jittering):
    def train_transforms(sample):
        # 调整图像尺寸
        sample = resize_sample(sample, image_shape=shape)
        # 调整尺度
        sample = spatial_augment_sample(sample)
        # 转换为tensor类型
        sample = to_tensor_sample(sample)
        # 应用 Homography Adaptation 图像增强
        sample = ha_augment_sample(sample, jitter_paramters=jittering)
        return sample

    return {'train': train_transforms}


class GetData(Dataset):
    def __init__(self, config, transforms=None):
        """
        Get the list containing all images and labels.
        """
        datafile = open(config.train_txt, 'r')
        # 读取文件中的每一行，将所有行作为字符串列表返回，行尾的换行符（\n）也包含在内
        lines = datafile.readlines()

        dataset = []
        for line in lines:
            # 用于移除字符串末尾的空白字符（包括换行符\n）
            line = line.rstrip()
            # 以空白字符（包括空格、制表符等）为分隔符，将每一行文本分割成一个字符串列表data，这里只有一个元素
            data = line.split()
            dataset.append(data[0])

        self.config = config
        self.dataset = dataset
        self.root = config.train_root

        self.transforms = transforms

    def __getitem__(self, index):
        """
        Return image'data and its label.
        """
        # 获取训练图片
        img_path = self.dataset[index]
        img_file = self.root + img_path
        img = Image.open(img_file)

        # image.mode == 'L' means the image is in gray scale 
        if img.mode == 'L':
            # 将灰度图转换为RGB
            img_new = Image.new("RGB", img.size)
            img_new.paste(img)
            sample = {'image': img_new, 'idx': index}
        else:
            sample = {'image': img, 'idx': index}

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        """
        Return the number of all data.
        """
        return len(self.dataset)


def get_data_loader(
        config,
        transforms=None,
        sampler=None,
        drop_last=True,
):
    """
    Return batch data for training.
    """
    transforms = image_transforms(shape=config.image_shape, jittering=config.jittering)
    dataset = GetData(config, transforms=transforms['train'])

    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        # 是否丢弃最后一个不完整的批次
        drop_last=drop_last
    )

    return train_loader
