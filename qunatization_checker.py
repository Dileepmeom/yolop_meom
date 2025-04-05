import onnx
from onnx import helper

# Load your ONNX model
# model = onnx.load('./weights/yolop-320-320.onnx')
model = onnx.load('yolop-320-320_quantized.onnx')

# Check for quantization nodes in the graph
quantization_nodes = [node for node in model.graph.node if node.op_type in ['QuantizeLinear', 'DequantizeLinear', 'QLinearConv', 'QLinearMatMul']]
if quantization_nodes:
    print("The model appears to be quantized. Found quantization nodes:")
    for node in quantization_nodes:
        print(f" - {node.op_type} (name: {node.name})")
else:
    print("No quantization nodes found. The model may not be quantized.")

# Optionally, inspect initializers to check their data types
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

print("\nModel initializers data types:")
for initializer in model.graph.initializer:
    np_dtype = TENSOR_TYPE_TO_NP_TYPE[initializer.data_type]
    print(f"{initializer.name}: {np_dtype}")
