# 모델을 불러오는 파일 => S3 버킷에 올라가있는 json 형태의 모델 파일을 다운 받는 형식
# MobileNet v2 => 이미지 구분 모델

import tensorflow as tf

def load_model():
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    print('MovileNetV2 model successfully loaded ...')
    return model

model = load_model()