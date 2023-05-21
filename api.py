from fastapi import FastAPI, Form, File
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from test_custom import run_custom_test
from typing import Annotated
import os

CLASS_NAMES = {3: "Pituitary Tumor detected", 0: "No tumor detected",
               2: "Meningioma Tumor", 2: "Glioma Tumor detected"}

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
    result = CLASS_NAMES[result]

    if os.path.exists("test-images/BT/image.jpg"):
        os.remove("test-images/BT/image.jpg")

    return {"data": result}
