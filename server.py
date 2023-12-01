import fastapi
import torch
import base64
import numpy
import cv2
import minimodel2
from io import BytesIO
from simple_eval import create_image
from fastapi.responses import StreamingResponse

app = fastapi.FastAPI()


@app.on_event("startup")
async def startup_event():
  app.state.model = minimodel2.miniModel(image_width=448, image_height=448, image_channels=1, grid_ratio=64, num_bounding_boxes=1, num_classes=10, dropout=0.4)
  app.state.model.load_state_dict(torch.load("./yolo_2_299.pth"))
  app.state.model.eval()

@app.on_event("shutdown")
async def startup_event():
  print("Shutting down")


@app.get("/")
def on_root():
  return { "message": "Hello App" }


@app.post("/one_number")
async def one_number(request: fastapi.Request):
    raw = (await request.json())["img"]
    raw = raw.split(',')[1]
    npArr = numpy.frombuffer(base64.b64decode(raw), numpy.uint8)
    img = cv2.imdecode(npArr, cv2.IMREAD_COLOR)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (448, 448), interpolation=cv2.INTER_LINEAR)
    npImg = numpy.expand_dims(grayImage, axis=0)
    npImgTensor = torch.tensor(npImg)
    npImgTensor = npImgTensor.unsqueeze(dim=0).float()
    npImgTensor = npImgTensor.view(1, 1, 448, 448)
    prediction = app.state.model(npImgTensor)

    plt = create_image(image=npImgTensor.squeeze(0), target=None, prediction=prediction.squeeze(0))
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64

