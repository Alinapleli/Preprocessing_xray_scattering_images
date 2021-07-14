import numpy as np
import torch


class PatchLoader(object):
    def __init__(self, imgs: list, labels: list, batch_size: int, bins: int = bin_size):
        self.batch_size = batch_size
        self.bins = bins
        self.imgs = imgs
        self.labels = labels
        self._size = len(labels)
        self._indices = np.arange(len(labels))

    @classmethod
    def from_data(cls, data, batch_size: int, bins: int = bin_size):
        imgs = [d[0] for d in data]
        labels = [d[2] for d in data]
        return cls(imgs, labels, batch_size, bins)

    def _get_batch(self, idx):
        bins = self.bins
        imgs = [self.imgs[i] for i in idx]
        labels = [self.labels[i] for i in idx]

        sizes = np.array([img.shape for img in imgs])

        points = (np.random.uniform(0, 0.5, sizes.shape) * sizes).astype(np.int)

        imgs = [img[point[0]:point[0] + size[0] // 2, point[1]:point[1] + size[1] // 2]
                for img, point, size in zip(imgs, points, sizes)
                ]
        labels = [scale(img, label) for img, label in zip(imgs, labels)]
        hists = [normalize(np.log(np.histogram(img, bins=bins)[0] + 1))
                 for img in imgs]

        return (
            torch.tensor(hists, dtype=torch.float32, device='cuda'),
            torch.tensor(labels, dtype=torch.float32, device='cuda')
        )

    def test_iteration(self):
        for i in range(len(self)):
            idx = self._indices[i:i + self.batch_size]
            yield self.get_batch_with_images(idx)

    def get_batch_with_images(self, idx):

        bins = self.bins
        imgs = [self.imgs[i] for i in idx]
        labels = [self.labels[i] for i in idx]

        sizes = np.array([img.shape for img in imgs])

        points = (np.random.uniform(0, 0.5, sizes.shape) * sizes).astype(np.int)

        imgs = [img[point[0]:point[0] + size[0] // 2, point[1]:point[1] + size[1] // 2]
                for img, point, size in zip(imgs, points, sizes)
                ]
        labels = [scale(img, label) for img, label in zip(imgs, labels)]
        hists = [normalize(np.log(np.histogram(img, bins=bins)[0] + 1))
                 for img in imgs]

        return (
            torch.tensor(hists, dtype=torch.float32, device='cuda'),
            torch.tensor(labels, dtype=torch.float32, device='cuda'),
            imgs
        )

    def __len__(self):
        return self._size

    def __iter__(self):
        np.random.shuffle(self._indices)

        for i in range(len(self)):
            yield self._get_batch(self._indices[1 + (i * self.batch_size):(i * self.batch_size) + self.batch_size])


def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def scale(img, labels):
    return (np.asarray(labels) - img.min()) / (img.max() - img.min())