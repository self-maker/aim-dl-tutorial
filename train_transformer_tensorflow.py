import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import DataCollatorWithPadding
from datasets import load_dataset
import numpy as np

# 데이터셋 로드
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 데이터 전처리 함수
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

# 데이터셋 토큰화 및 전처리
tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets.set_format(type='tensorflow', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

# 데이터셋을 TensorFlow 형식으로 변환
train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=['input_ids', 'token_type_ids', 'attention_mask'],
    label_cols=['label'],
    shuffle=True,
    batch_size=8,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf"),
)

eval_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=['input_ids', 'token_type_ids', 'attention_mask'],
    label_cols=['label'],
    shuffle=False,
    batch_size=8,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf"),
)

# 모델 로드
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 옵티마이저, 손실 함수, 평가 지표 설정
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# 모델 학습
model.fit(train_dataset, validation_data=eval_dataset, epochs=3)

# 모델 평가
model.evaluate(eval_dataset)