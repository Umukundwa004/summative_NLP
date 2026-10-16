# Kinyarwanda news topic classifier

Classify a news text in **Kinyarwanda** into a topic (politics, sport, economy, health, ...).

```
notebook/   the experiments (data, cleaning, models, evaluation, error analysis, saving the model)
models/     the trained Kinyarwanda model, written by the notebook (kinnews)
api/        FastAPI server: loads models/, answers /predict, and serves the web page
frontend/   index.html: the page where the user types the text
data/       the dataset (downloaded by the notebook; do not upload it to git)
results/    tables and error files written by the notebook
```

How the parts work together:

```
 notebook  --saves-->  models/        FastAPI (api/main.py)  --loads-->  models/
                                      GET /          -> frontend/index.html  (the page in the browser)
 browser page  -- POST /predict {"text": "...", "language": "kinnews"} -->  FastAPI  -->  model
               <-- {"label": "sport", "confidence": 0.93, "probabilities": {...}} --
```

## 1. Train the model (notebook)

1. Open `notebook/kinnews_experiments.ipynb` and run all cells. The KINNEWS dataset is downloaded into `data/` automatically (no account needed).
2. Run the settings cell with `DATASET = "kinnews"`.
3. Section 10 saves the chosen model into `models/<dataset>/`, writes `api/nlp_utils.py` and `api/requirements.txt`, and checks that the server's model gives the same predictions as the notebook.

> The experiments with the BiLSTM are slow on a CPU for the large Kinyarwanda dataset. Use a free Colab GPU (and download the `models/` and `api/` folders afterwards), or lower `MAX_LEN` and the number of epochs.

## 2. Start the server and open the page

```bash
pip install -r api/requirements.txt
uvicorn main:app --app-dir api --port 8000
```

* **Web page:** <http://localhost:8000>
* **Interactive API documentation:** <http://localhost:8000/docs>
* Quick test from the terminal:
  ```bash
  curl http://localhost:8000/languages
  curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
       -d '{"text": "Ikipe y igihugu yatsinze umukino", "language": "kinnews"}'
  ```
* Tests: `pip install pytest httpx` then `pytest api -q`

| Endpoint | What it does |
|---|---|
| `GET /` | the web page |
| `GET /health` | is the server running, which models are loaded |
| `GET /languages` | the Kinyarwanda model, its topics and test scores |
| `POST /predict` | body `{"text": "...", "language": "kinnews"}` (language is optional); returns the topic, the confidence and the probability of every topic |

Errors are clear: `404` unknown language, `422` empty or too long text, `503` no model loaded yet.

## Notes

* The saved models can be large (especially a Transformer). Add `data/`, `models/` and `.venv/` to `.gitignore`.
* To serve to other computers: `uvicorn main:app --app-dir api --host 0.0.0.0 --port 8000`.
* The server sends `Access-Control-Allow-Origin: *` so `frontend/index.html` also works when opened from your disk (type the server address in the "Server address (advanced)" box). Set the environment variable `ALLOWED_ORIGINS` to restrict it. To use another models folder, set `MODELS_DIR`.
* The model can make mistakes. It is a study project, not a tool for important decisions.
