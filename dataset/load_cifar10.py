import torch
import torchvision
import torchvision.transforms as transforms

# torchvision 데이터셋은 0~1 범위를 가진 PILImage임.
# -1, 1 로 Normalize 처리를 해줘야함.
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 4

# abstract class인 Dataset 클래스 타입임
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train='True',
    download=True,
    transform=transform
    )
