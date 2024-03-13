import tensorflow as tf
from tensorflow.keras import layers, models, datasets, utils, optimizers, losses

# 1. 데이터셋 로드 및 전처리
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 이미지를 [0, 1] 범위로 정규화합니다.
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 레이블을 원-핫 인코딩합니다.
train_labels = utils.to_categorical(train_labels)
test_labels = utils.to_categorical(test_labels)

# 2. CNN 모델 정의
model = models.Sequential([
    layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 3. 손실 함수와 옵티마이저 정의
model.compile(optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. 신경망 학습
model.fit(train_images, train_labels, epochs=10, batch_size=64)

# 5. 테스트 데이터에 대한 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Accuracy of the network on the 10000 test images: {test_acc * 100:.2f} %')

# 학습된 모델 저장
model.save('cnn_model_tensorflow.h5')
