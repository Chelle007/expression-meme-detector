"""
One-time script: convert emotion_model.hdf5 (Keras) to emotion_model.onnx.
Run once with: pip install tensorflow tf2onnx onnx  (or use a separate venv).
Then run main.py with onnxruntime only.
"""

import tensorflow as tf
import tf2onnx
import onnx

HDF5_PATH = "emotion_model.hdf5"
ONNX_PATH = "emotion_model.onnx"

def main():
    print(f"Loading {HDF5_PATH}...")
    model = tf.keras.models.load_model(HDF5_PATH, compile=False)
    # Input: (batch, 64, 64, 1) grayscale
    input_shape = model.input_shape
    input_spec = [tf.TensorSpec([None] + list(input_shape[1:]), tf.float32, name="input")]
    print(f"Converting to ONNX (input shape {input_shape})...")
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_spec, opset=13)
    onnx.save(onnx_model, ONNX_PATH)
    print(f"Saved {ONNX_PATH}")

if __name__ == "__main__":
    main()
