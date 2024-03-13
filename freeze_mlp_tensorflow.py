import tensorflow as tf
import tf2onnx
import onnx

# 미리 훈련된 Keras 모델을 로드합니다.
model = tf.keras.models.load_model('./mlp_model_tensorflow.h5')

# 모델을 ONNX로 변환하기 위한 준비 작업
spec = (tf.TensorSpec((None, 28, 28, 3), tf.float32, name='input'),)

# tf2onnx를 사용하여 모델을 ONNX로 변환
output_path = 'mlp_model_tensorflow.onnx'  # 출력될 ONNX 파일의 경로
model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

print(f'Model has been converted to ONNX and saved to {output_path}')