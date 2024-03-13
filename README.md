# AIM-DL-Tutorial
이 튜토리얼은 딥러닝 학습 프레임워크 동작에 대한 자료를 제공합니다. Miniconda Python 환경을 기준으로 하고 있습니다.

## Miniconda 
- **설명**: 파이썬 패키지와 가상환경을 관리하는 도구로, Anaconda의 축소 버전입니다.
- **특징**: Anaconda 대비 가볍고 필요한 기능만 포함.
- **설치**: [Miniconda 설치 가이드](https://docs.anaconda.com/free/miniconda/miniconda-install/) 참조.

## Anaconda 환경 설치
```bash
conda create –n aimdl python=3.9
conda activate aimdl
```

## PyTorch 설치
```bash
pip install torch torchvision
```

## TensorFlow 설치
```bash
pip install tensorflow tf-keras
```

## MNIST Dataset 
- 손글씨(0~9) 이미지 분류 데이터인 MNIST Dataset을 사용하여 학습을 진행 합니다.
- 60,000개의 트레이닝 셋과 10,000개의 테스트 셋으로 구성되어 있습니다.
- 0에서 1까지의 값을 갖는 고정 크기 이미지 (28x28 픽셀)로 구성되어 있습니다.



## 모델 ONNX 변환 코드

### MLP 모델 ONNX 변환 코드
- `freeze_mlp_pytorch.py`: PyTorch 기반 MLP 모델을 ONNX로 변환하는 스크립트
- `freeze_mlp_tensorflow.py`: TensorFlow 기반 MLP 모델을 ONNX로 변환하는 스크립트

### CNN 모델 ONNX 변환 코드
- `freeze_cnn_pytorch.py`: PyTorch 기반 CNN 모델을 ONNX로 변환하는 스크립트
- `freeze_cnn_tensorflow.py`: TensorFlow 기반 CNN 모델을 ONNX로 변환하는 스크립트

