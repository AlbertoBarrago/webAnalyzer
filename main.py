import os
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
app = FastAPI()

# Static template setup
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Hugging face setup
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
local_dir = "./models/sentiment"
sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)

logger.info("Hugging Face model loaded and ready!")

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
