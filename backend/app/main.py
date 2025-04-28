from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os
from app.utils import process_image

app = FastAPI()

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    file_location = f"app/temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        texts, output_path = process_image(file_location)
        return JSONResponse(content={"texts": texts, "image_url": "/result_image"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.remove(file_location)

@app.get("/result_image")
def get_result_image():
    return FileResponse("app/output.png", media_type="image/png")
