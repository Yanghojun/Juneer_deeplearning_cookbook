import onnx
import onnxruntime as ort

if __name__ == '__main__':
    onnx_model = onnx.load('./weights/end2end.onnx')
    onnx.checker.check_model(onnx_model)