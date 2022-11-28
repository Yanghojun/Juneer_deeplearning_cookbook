import sys
import os
sys.path.append(os.getcwd())
import onnx
import onnxruntime as ort
import torch
from models.simple_cnn import SimpleCNN
from data.load_mnist import load_mnist
import numpy as np

def export_onnx(model:torch.nn, save_path:str):
    # 아래 예시는 배치사이즈 1, Fashion MNIST 데이터를 학습한 pytorch 모델에 대한 것임
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)

    # onnx 모델의 가중치를 접근하기위해 name specifying이 가능한것으로 보임
    input_names = ['actual_input_1'] + [f'learned_{i}' for i in range(20)]
    output_names = ['output1']

    torch.onnx.export(
        model, 
        dummy_input,
        os.path.join(save_path, 'ts_mn.onnx'),
        verbose=True,       # export 할 때 사람이 읽을 수 있도록 print문으로 콘솔창에 출력
        input_names=input_names, 
        output_names=output_names
        )

def validate_onnx(path:str)->None:
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph)[:20000])

def create_session(onnx_path:str):
    ort_sess = ort.InferenceSession(onnx_path)
    outputs = ort_sess.run(
        None,
        {'input':np.random.randn(1,3,224,224).astype(np.float32)},  # 여기서는 numpy를 사용해야 한다고 함
    )
    print(len(outputs))
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)
    # x, y = 
    # outputs = ort_sess.run(None, {'input': x.numpy()})

if __name__ == '__main__':
    # load and validate onnx model
    # validate_onnx('./weights/end2end.onnx')

    # load mnist data
    # imgs, labels = load_mnist('./data/MNIST/train-images-idx3-ubyte', './data/MNIST/train-labels-idx1-ubyte')
    
    # create session and do inference    
    # ort_sess = ort.InferenceSession('./weights/end2end.onnx')

    # model = SimpleCNN()
    # model.load_state_dict(torch.load('./weights/simple_cnn_fashion_mnist.pt'))
    # model.eval()
    
    # export_onnx(model=model, save_path='./weights')
    # validate_onnx('./weights/ts_mn.onnx')
    # onnx_model = onnx.load('./weights/ts_mn.onnx')
    create_session('./weights/end2end.onnx')
    # validate_onnx('./weights/end2end.onnx')