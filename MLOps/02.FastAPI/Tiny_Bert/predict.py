from PIL.Image import Image # pip install pillow
import tensorflow as tf
import numpy as np
from model_loader import model

def predict(image: Image): # pydantic -> type check
    # img => 규격화(인풋 이미지의 사이즈 정규화)
    image = np.asarray(image.resize((224, 224)))[..., :3] # Framework => () 값을 입력을 많이받습니다. | RGB
    # print(f'Image first: {image}')

    image = np.expand_dims(image, 0)
    # print(f'Image second: {image}')

    image = image / 127.5 - 1.0 # Scaler => -1 ~ 1 사이의 값을 갖게 됩니다.
    # print(f'Image third: {image}') # IO => 영상까지도 건들게 되겠죠.
    
    results = tf.keras.applications.imagenet_utils.decode_predictions(
        model.predict(image),
        1 # 결과 예측 갯수
    )[0]
    
    print(results) # [[('n02112018', 'Pomeranian', 0.45100003), ('n02112137', 'chow', 0.027857592), ('n02113624', 'toy_poodle', 0.024156988)]]

    result_list = []
    for i in results:
        result_list.append({
            'Class' : i[1],
            'Confidence' : f'{i[2]*100:0.2f}%'
        })

    return result_list

# 저는 debug=True 옵션이 없다고 떠서 app = FastAPI(debug=True) 로 변경했습니다.