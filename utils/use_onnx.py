import onnx
import onnxruntime as ort
from data.load_mnist import load_mnist

def validate_onnx(path:str)->None:
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)

def create_session(path:str):
    ort_sess = ort.InferenceSession(path)

    x, y = 
    outputs = ort_sess.run(None, {'input': x.numpy()})

if __name__ == '__main__':
    # load and validate onnx model
    validate_onnx('./weights/end2end.onnx')

    # load mnist data
    imgs, labels = load_mnist('./data/MNIST/train-images-idx3-ubyte', './data/MNIST/train-labels-idx1-ubyte')
    
    # create session and do inference    
    ort_sess = ort.InferenceSession('./weights/end2end.onnx')
