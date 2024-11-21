from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from PIL import Image
from predict import predict
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def show_item(item_id):
    return {"item_id": item_id}

@app.post('/api/v1/predict')
# @app.post('/api/v2/predict') # 버전 관리
async def img_predict(file: UploadFile = File(...)):
    raw_data = await file.read()
    print(raw_data)

    image_bytes_io = BytesIO(raw_data)
    print(image_bytes_io)

    img = Image.open(image_bytes_io)
    pred = predict(img)

    return pred