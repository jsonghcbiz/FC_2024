# tensorflow에서 모바일넷V2
import tensorflow as tf

def load_model():
    model = tf.keras.applications.MobileNetV2(weights='imagenet') # 이미지 모델
    print('Model Successfully Loaded....')

    return model

model = load_model()