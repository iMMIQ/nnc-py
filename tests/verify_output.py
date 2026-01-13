"""Verify C code output against ONNX Runtime."""
import numpy as np
import onnxruntime as ort

# Load the ONNX model
model_path = "simple_conv.onnx"
session = ort.InferenceSession(model_path)

# Get input info
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

print(f"Input name: {input_name}")
print(f"Input shape: {input_shape}")

# Create test input data (same as test_runner.c)
# test pattern: input[i] = i * 0.01
size = np.prod(input_shape)
input_data = np.arange(size, dtype=np.float32) * 0.01
input_data = input_data.reshape(input_shape)

print(f"\nInput data (first 10):")
print(input_data.flatten()[:10])

# Run inference
outputs = session.run(None, {input_name: input_data})
output = outputs[0]

print(f"\nOutput shape: {output.shape}")
print(f"Output data (first 10):")
print(output.flatten()[:10])

# Save to file for comparison
np.save("onnx_output.npy", output)
print("\nSaved ONNX output to onnx_output.npy")
