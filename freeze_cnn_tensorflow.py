import tensorflow as tf
import tf2onnx
import onnx

# 미리 훈련된 Keras 모델을 로드합니다.
model = tf.keras.models.load_model('./cnn_model_tensorflow.h5')

# 모델을 ONNX로 변환하기 위한 준비 작업
spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name='input'),)

# tf2onnx를 사용하여 모델을 ONNX로 변환
output_path = 'cnn_model_tensorflow.onnx'  # 출력될 ONNX 파일의 경로
model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

print(f'Model has been converted to ONNX and saved to {output_path}')

import tensorflow as tf
import keras2onnx
import onnx

# 미리 훈련된 Keras 모델을 로드합니다.
model = tf.keras.models.load_model('./cnn_model_tensorflow.h5')

# 모델을 ONNX로 변환
onnx_model = keras2onnx.convert_keras(model, model.name)

# ONNX 모델 저장
output_path = "cnn_model_tensorflow.onnx"
onnx.save_model(onnx_model, output_path)

print(f"Model has been converted to ONNX and saved to {output_path}")


