from fastapi import FastAPI, Form, File
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from test_custom import run_custom_test
from typing import Annotated
import os

CLASS_NAMES = {
    0: "No tumor detected",
    1: "Glioma Tumor detected",
    2: "Meningioma Tumor",
    3: "Pituitary Tumor detected"
}

COLORS = {
    0: 'green',
    1: 'red',
    2: 'orange',
    3: 'orangered'
}

app = FastAPI()
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
templates = Jinja2Templates(directory="ui")


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api")
async def root(
    image: Annotated[bytes, File()],
    # name:  Annotated[str, Form()]
):
    with open('test-images/BT/image.jpg', 'wb') as img:
        img.write(image)
        img.close()

    result = run_custom_test(
        image='image.jpg', device='cpu', dataset="BT", batch_size=32)
    # print(result)
    
    color = COLORS[result]
    result = CLASS_NAMES[result]

    if os.path.exists("test-images/BT/image.jpg"):
        os.remove("test-images/BT/image.jpg")

    return {"data": result, 'color': color}
