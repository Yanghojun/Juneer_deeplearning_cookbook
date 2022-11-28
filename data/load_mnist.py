import torch
import torchvision
import numpy as np
import os
import matplotlib.pyplot as plt
from struct import unpack
import matplotlib.cm as cm
from typing import Tuple

def output_label(label):
    output_mapping = {
        0: "T-shirt/Top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat", 
        5: "Sandal", 
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot"
    }

    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

def load_mnist(path:str, kind:str='train') -> Tuple[np.uint8, np.uint8]:
    if kind == 'test':
        kind = 't10k'
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return (images, labels)


if __name__ == '__main__':
    # download 지원을 안하게됨
    # imagenet_data = torchvision.datasets.ImageNet('/data/MNIST/',download=True)

    imgs, labels = load_mnist('./data/MNIST', kind='train')
    print(imgs[0], labels[0])