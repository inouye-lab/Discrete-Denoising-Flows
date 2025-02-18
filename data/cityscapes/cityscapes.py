import os
from collections import namedtuple

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])
classes = [
    CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

map_id_to_category_id = [x.category_id for x in classes]
map_id_to_category_id = torch.tensor(map_id_to_category_id)

ROOT = os.path.dirname(os.path.abspath(__file__))


class Cityscapes(data.Dataset):
    def __init__(self, root=ROOT, split='train', resolution=(32, 64)):

        H, W = resolution

        self.root = os.path.expanduser(root)
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.Pad((1, 2), padding_mode='edge'),
                                             transforms.RandomCrop((32, 64))])
        self.split = split

        if split not in ('train', 'val', 'test'):
            raise ValueError('split should be one of {train, val, test}')

        if not self._check_exists(H, W):
            raise RuntimeError('Dataset not found (or incomplete) at {}'.format(self.root))

        self.data = torch.from_numpy(
            np.load(os.path.join(self.root, 'preprocessed', split + '_32x64.npy')))

    def __getitem__(self, index):
        img = self.data[index]

        img = img.long()
        img = map_id_to_category_id[img]

        assert img.size(0) == 1
        img = img[0]
        img = Image.fromarray(img.numpy().astype('uint8'))
        img = self.transform(img)
        img = np.array(img)
        img = torch.tensor(img).long()
        img = img.unsqueeze(0)

        return img

    def __len__(self):
        return len(self.data)

    def _check_exists(self, H, W):
        train_path = os.path.join(self.root, 'preprocessed', 'train_32x64.npy')
        val_path = os.path.join(self.root, 'preprocessed', 'val_32x64.npy')
        test_path = os.path.join(self.root, 'preprocessed', 'test_32x64.npy')

        return os.path.exists(train_path) and os.path.exists(val_path) and \
               os.path.exists(test_path)


def get(batch_size, root=ROOT):
    train_set = Cityscapes(root=root, split='train')
    val_set = Cityscapes(root=root, split='val')
    test_set = Cityscapes(root=root, split='test')

    trainloader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=4,
                                              drop_last=True)

    valloader = torch.utils.data.DataLoader(val_set,
                                            batch_size=batch_size,
                                            shuffle=False, num_workers=4)

    testloader = torch.utils.data.DataLoader(test_set,
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=4)

    return trainloader, valloader, testloader
