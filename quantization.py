from onnxruntime.quantization import quantize_dynamic, QuantType

# Specify your original model and output quantized model paths
original_model = './weights/yolop-320-320.onnx'
quantized_model = "yolop-320-320_quantized.onnx"

# Perform dynamic quantization; here, weights will be quantized to int8
quantize_dynamic(original_model, quantized_model, weight_type=QuantType.QInt8)

print(f"Quantized model saved to: {quantized_model}")
