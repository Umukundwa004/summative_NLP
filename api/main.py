"""FastAPI server: predicts the topic of Kinyarwanda / Kirundi news text AND serves the web page.

Run it from the project folder:

    pip install -r api/requirements.txt
    uvicorn main:app --app-dir api --port 8000

Then open  http://localhost:8000          the web page (frontend/index.html)
           http://localhost:8000/docs     interactive documentation of the API

The models (folder models/) and api/nlp_utils.py are written by the notebook (Section 10).
Every folder models/<dataset>/ (kinnews = Kinyarwanda, kirnews = Kirundi) becomes one language.
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

API_DIR = Path(__file__).resolve().parent         # <project>/api
ROOT_DIR = API_DIR.parent                         # <project>
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT_DIR / "models"))   # <project>/models  
FRONTEND_DIR = ROOT_DIR / "frontend"              # <project>/frontend (the HTML page)
sys.path.insert(0, str(API_DIR))                  # so that "import nlp_utils" works

# dataset key ("kinnews", "kirnews") -> loaded classifier
classifiers: Dict[str, object] = {}


# ── Loading the models once, when the server starts ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    classifiers.clear()
    try:
        from nlp_utils import DeployedClassifier          # written by the notebook
    except ImportError:
        print("  api/nlp_utils.py not found. Run the notebook (Section 10) first.")
    else:
        for config_file in sorted(MODELS_DIR.glob("*/config.json")):
            classifier = DeployedClassifier(config_file.parent)
            classifiers[classifier.config["dataset"]] = classifier
            print(f" Loaded {classifier.config['language']} model ({classifier.config['dataset']})")
        if not classifiers:
            print(f"  No models found in {MODELS_DIR}. Run the notebook (Section 10) first.")
    yield
    classifiers.clear()


app = FastAPI(
    title="Kinyarwanda & Kirundi news classifier",
    description="Send a news text in Kinyarwanda or Kirundi and get its predicted topic.",
    version="1.0.0",
    lifespan=lifespan,
)

# The HTML page is served by this same server, so it needs no CORS. CORS is only for pages opened from elsewhere
# (for example index.html opened from your disk). Set ALLOWED_ORIGINS to restrict it.
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


# ── Request and response formats ───────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000, description="The news text (title and/or article).")
    language: Optional[str] = Field(
        None,
        description="Which language model to use: 'kinnews' or 'Kinyarwanda', 'kirnews' or 'Kirundi'. "
                    "If you leave it out, the first available model is used.")


class PredictResponse(BaseModel):
    dataset: str = Field(..., description="'kinnews' or 'kirnews'")
    language: str = Field(..., description="'Kinyarwanda' or 'Kirundi'")
    label: str = Field(..., description="The predicted topic")
    confidence: float = Field(..., description="Probability of the predicted topic (0 to 1)")
    probabilities: Dict[str, float] = Field(..., description="Probability of every topic, highest first")
    cleaned_text: str = Field(..., description="The text after cleaning (what the model really saw)")


class LanguageInfo(BaseModel):
    dataset: str
    language: str
    labels: List[str]
    best_model: str
    validation_macro_f1: float
    test_macro_f1: float
    test_accuracy: float


# ── Helper: find the classifier for the requested language ─────────────────
def find_classifier(requested: Optional[str]):
    if not classifiers:
        raise HTTPException(status_code=503,
                            detail="No models are loaded. Run the notebook (Section 10), then restart the server.")
    if requested is None or requested.strip() == "":
        return next(iter(classifiers.values()))
    wanted = requested.strip().lower()
    for key, classifier in classifiers.items():
        if wanted in (key.lower(), classifier.config["language"].lower()):
            return classifier
    available = ", ".join(f"{k} ({c.config['language']})" for k, c in classifiers.items())
    raise HTTPException(status_code=404, detail=f"Unknown language '{requested}'. Available: {available}")


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    # The web page. If the frontend folder is missing, show a small JSON message instead.
    page = FRONTEND_DIR / "index.html"
    if page.exists():
        return FileResponse(page, media_type="text/html")
    return {"name": app.title, "docs": "/docs", "health": "/health", "languages": "/languages", "predict": "POST /predict"}


@app.get("/health")
def health():
    # A quick check that the server is running and which models it has
    return {"status": "ok", "models_loaded": sorted(classifiers)}


@app.get("/languages", response_model=List[LanguageInfo])
def languages():
    # The languages (models) this server can classify, with their topics and test scores
    return [LanguageInfo(dataset=key, language=c.config["language"], labels=c.labels, best_model=c.config["best_model"],
                         validation_macro_f1=c.config["validation_macro_f1"], test_macro_f1=c.config["test_macro_f1"],
                         test_accuracy=c.config["test_accuracy"])
            for key, c in classifiers.items()]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Plain "def" (not async): FastAPI runs it in a worker thread, so the heavy model code does not block the server
    classifier = find_classifier(request.language)
    result = classifier.predict(request.text)
    if result is None:
        raise HTTPException(status_code=422, detail="The text is empty after cleaning. Please send some real text.")
    return PredictResponse(dataset=classifier.config["dataset"], language=classifier.config["language"], **result)
