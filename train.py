import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from data.load_mnist import load_mnist, output_label
from utils.preprocess import MNistDataset
from torch.utils.data import DataLoader
from models.simple_cnn import SimpleCNN

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Fashion mnist 학습 데이터 준비
    batch_size = 100
    train_mnist_dataset = MNistDataset(path='./data/MNIST', kind='train')       # 60,000장 이미지 및 레이블
    test_mnist_dataset = MNistDataset(path='./data/MNIST', kind='test')         # 10,000장 이미지 및 레이블
    train_loader = DataLoader(
        train_mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = SimpleCNN()
    model.to(device)

    error = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 5
    count = 0
    # Lists for visualization of loss and accuracy
    loss_list = []
    iteration_list = []
    accuracy_list = []

    # Lists for knowing classwise accuracy
    predcitions_list = []
    labels_list = []

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # gpu로 옮길 수 있으면 옮기기
            images, labels = images.to(device), labels.to(device)
            
            train = images.view(batch_size, 1, 28, 28)
            labels = labels

            # 포워드하고 error 구하기
            outputs = model(train)
            loss = error(outputs, labels)

            # gradient 0으로 초기화
            optimizer.zero_grad()

            # 역전파
            loss.backward()

            # 역전파로 전달된 값으로 각 파라미터 업데이트
            optimizer.step()

            count += 1

            if count % 50 == 0:
                total = 0
                correct = 0

                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    labels_list.append(labels)

                    test = images.view(batch_size, 1, 28, 28)
                    outputs = model(test)

                    predictions = torch.max(outputs, 1)[1].to(device)       # [1]이 인덱스
                    predcitions_list.append(predictions)
                    correct += (predictions == labels).sum()

                    total += len(labels)
                
                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

            if count % 500 == 0:
                print(f"Iteration: {count}, Loss: {loss.data}, Accuracy: {accuracy}")

    torch.save(model.state_dict(), './weights/simple_cnn_fashion_mnist.pt')