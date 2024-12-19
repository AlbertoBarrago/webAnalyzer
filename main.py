import io
import os

import numpy as np
import psutil
import torch
from PIL import Image
from starlette.responses import StreamingResponse

os.environ["HF_HOME"] = "./models/sentiment"
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.requests import Request
from fastapi import FastAPI, HTTPException
from app.utils import (
    logger,
    extract_text_from_url)
from transformers import pipeline
from diffusers import StableDiffusionPipeline
app = FastAPI()

# Static template setup
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Hugging face setup
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
local_dir = "./models/sentiment"
sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)

logger.info("Hugging Face model loaded and ready!")

# Text-to-image setup
model_id = "stabilityai/stable-diffusion-2-1-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)
pipe = pipe.to(device)

logger.info("Text to image ready to destroy your GPU")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/analyze/")
async def analyze_webpage(url: str):
    try:
        text = extract_text_from_url(url)
        logger.info(f"Extracted text length: {len(text)} characters")

        analysis = sentiment_analyzer(text[:512]) #limit 512 chars for avoiding errors with models
        logger.info(f"Analysis complete: {analysis}")

        return {"url": url, "analysis": analysis}
    except Exception as e:
        logger.error(f"Error analyzing URL: {e}")
        raise HTTPException(status_code=400, detail=str(e))



@app.get("/text-to-image/")
async def text_to_image(text: str):
    try:
        logger.info(f"Generating image for text: {text}")
        output = pipe(text)
        image = output.images[0]

        if isinstance(image, torch.Tensor):
            image_array = (image.cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image_array)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate image from text")


@app.get("/status/")
async def system_status():
    mem = psutil.virtual_memory()
    ram_available = mem.available / (1024 ** 3)
    ram_total = mem.total / (1024 ** 3)

    cpu_load = psutil.cpu_percent(interval=1)

    return {
        "cpu_load_percent": cpu_load,
        "ram_available_gb": round(ram_available, 2),
        "ram_total_gb": round(ram_total, 2),
        "can_run": ram_available > 2 and cpu_load < 80,
    }