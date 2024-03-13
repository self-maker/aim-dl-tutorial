import torch
from train_mlp_pytorch import MLP

PATH = './mlp_model_pytorch.ckpt'

# 모델을 ONNX로 변환하여 저장
dummy_input = torch.randn(1, 1, 28, 28) # 모델 입력 크기에 맞는 더미 데이터
model = MLP()
model.load_state_dict(torch.load(PATH))
model.eval()

onnx_path = "./mlp_model_pytorch.onnx"
torch.onnx.export(model, dummy_input, onnx_path, verbose=False)

print('Model is converted to ONNX format and saved.')


