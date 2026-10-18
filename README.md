# Kinyarwanda News Topic Classifier

An end-to-end natural language processing project for classifying a Kinyarwanda
news article into one of 14 topics. The project includes reproducible
experiments, a saved production model, a Streamlit web application, and an
optional FastAPI service.

**Try the deployed application:**
[Open the Kinyarwanda news classifier](https://summativenlp-mvcz2sv7jk56kouu45pdmg.streamlit.app/)

[youtube video](https://youtu.be/AlPnrd_C270)

**Other resources**

- [KINNEWS dataset on Hugging Face](https://huggingface.co/datasets/andreniyongabo/kinnews_kirnews)
- [KINNEWS source repository](https://github.com/saradhix/kinnews_kinnews)

## Problem

News publishers produce articles in Kinyarwanda across many subject areas.
Automatically identifying the topic can support search, filtering, content
organisation, and downstream analysis. This project predicts the topic of a
Kinyarwanda news title or article from the following 14 labels:

`politics`, `sport`, `economy`, `health`, `entertainment`, `history`,
`technology`, `tourism`, `culture`, `fashion`, `religion`, `environment`,
`education`, and `relationship`.

The system is a multi-class single-label classifier: each input receives one
topic, a confidence score, and probabilities for all topics. It is an
educational NLP system and should not be used as the sole basis for important
editorial, financial, medical, or other decisions.

## Dataset

The project uses **KINNEWS**, a labelled Kinyarwanda news classification
dataset. The original dataset was collected from Rwandan news websites and
newspapers. Its raw files contain a class index, English and Kinyarwanda
labels, URL, title, and article content; the cleaned files contain the class,
title, and content.

The experiment run recorded in `results/kinnews/` contains:

- 10,924 usable articles after preprocessing;
- 14 topic classes;
- 70% training, 15% validation, and 15% test data using one shared
  stratified split;
- 1,639 held-out test articles;
- no empty, conflicting, duplicate, or low-frequency classes removed in this
  run.

The notebook can download the data from the linked source. Dataset files are
normally excluded from Git with `.gitignore`; keep them locally under
`data/KINNEWS/` when reproducing the experiments.

## Methodology

### Text preparation

The same cleaning logic is used during training and inference:

1. HTML entities are unescaped and Unicode is normalised.
2. Invisible characters, URLs, and email addresses are removed.
3. Text is lower-cased and repeated whitespace is collapsed.
4. Empty, duplicate, and conflicting records are removed before splitting.

The deployed classifier returns the cleaned text in its prediction response so
that users can inspect what the model received.

### Candidate models

All experiments use the same cleaned data and split. The notebook evaluates:

- a majority-class baseline;
- TF-IDF word unigrams and bigrams with Logistic Regression;
- Complement Naive Bayes with TF-IDF unigrams and bigrams;
- TextCNN with class weights;
- BiLSTM with attention;
- BiLSTM with attention initialised with Word2Vec embeddings; and
- an ensemble that averages selected model probabilities.

Validation macro-F1 is used to select the final model. The test set is used
after model selection for the final comparison.

### Selected model

The deployed model is **TF-IDF word unigrams and bigrams plus balanced
Logistic Regression**. Its recorded settings are:

- n-gram range: `(1, 2)`;
- class weights: `balanced`;
- regularisation `C`: `1.0`; and
- maximum features: `300,000`.

The saved bundle is in [`models/kinnews/`](C:/Users/LENOVO/OneDrive/Desktop/machine%20learning/summative_NLP/models/kinnews/).
The application loads this bundle at startup; it does not retrain on every
request.

## Experiments and results

The final held-out test results are below. The primary comparison metric is
macro-F1 because it gives each topic equal weight despite class imbalance.

| Model | Macro-F1 | Weighted-F1 | Accuracy | Log loss |
|---|---:|---:|---:|---:|
| Logistic Regression | **0.7125** | 0.7705 | 0.7694 | 1.1597 |
| Ensemble | 0.6733 | 0.7496 | 0.7480 | 0.8797 |
| Complement Naive Bayes | 0.6479 | 0.7676 | **0.7779** | 1.0183 |
| BiLSTM + Word2Vec | 0.6365 | 0.7151 | 0.7108 | 0.9107 |
| TextCNN | 0.6197 | 0.6853 | 0.6852 | 0.9959 |
| BiLSTM + attention | 0.5583 | 0.6291 | 0.6223 | 1.4571 |
| Majority baseline | 0.0229 | 0.0612 | 0.1910 | 2.2433 |

The Logistic Regression model was selected because it achieved the best
validation macro-F1 (0.7074) and the best test macro-F1 (0.7125). Complement
Naive Bayes achieved slightly higher test accuracy, but its macro-F1 was lower,
which indicates weaker performance across the less frequent topics.

The per-topic results show strong performance on topics such as sport,
entertainment, religion, and relationship. Smaller classes such as education,
environment, culture, and history are more difficult, partly because they have
fewer test examples. Long articles can also be difficult because neural
experiments truncate their input to a maximum sequence length.

## Repository structure

```text
.
├── api/
│   ├── main.py              FastAPI server and API endpoints
│   ├── nlp_utils.py         Shared model loading, cleaning, and prediction
│   ├── requirements.txt     API dependencies
│   └── test_api.py          API tests
├── data/KINNEWS/            Raw and cleaned dataset files
├── frontend/index.html      Optional HTML client for FastAPI
├── models/kinnews/          Saved Logistic Regression model and metadata
├── notebook/                Training, experiments, evaluation, and export
├── results/kinnews/         Metrics and error-analysis outputs
├── screenshots/             Desktop and phone UI screenshots
├── streamlit_app.py         Deployed and local Streamlit entry point
└── requirements.txt         Application and notebook dependencies
```

## Run locally

Python 3.10 or newer is recommended.

### Streamlit web application

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, normally
`http://localhost:8501`. The app loads the saved model from
`models/kinnews/` and displays the predicted topic, confidence,
class-probability bars, and cleaned input text.

### FastAPI service

Install the API dependencies and start the server:

```bash
python -m pip install -r api/requirements.txt
uvicorn main:app --app-dir api --port 8000
```

Useful endpoints:

| Endpoint | Description |
|---|---|
| `GET /` | Serves the optional HTML client |
| `GET /health` | Reports server status and loaded models |
| `GET /languages` | Lists labels and recorded model scores |
| `POST /predict` | Classifies a Kinyarwanda text |
| `GET /docs` | Opens the interactive OpenAPI documentation |

Example request:

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Ikipe y'igihugu yatsinze umukino ejo mu kibuga.\",\"language\":\"kinnews\"}"
```

On macOS/Linux, replace the `^` line continuations with `\`.
`language` is optional. The response includes `dataset`, `language`, `label`,
`confidence`, probabilities for every topic, and `cleaned_text`.

Run the API tests from the repository root:

```bash
python -m pip install pytest httpx
pytest api -q
```

## Reproduce training and experiments

1. Open [`notebook/kinnews_NLP_classification.ipynb`](C:/Users/LENOVO/OneDrive/Desktop/machine%20learning/summative_NLP/notebook/kinnews_NLP_classification.ipynb)
   in Jupyter or Google Colab.
2. Install the notebook packages from the notebook's setup cell.
3. Set `DATASET = "kinnews"` and run the cells in order.
4. The notebook downloads/prepares KINNEWS, performs the shared split,
   trains each candidate, evaluates validation and test performance, and
   writes artifacts to `results/kinnews/`.
5. The model-export section writes the selected model to
   `models/kinnews/` and generates the shared inference utilities used by the
   API and Streamlit application.

The neural-network experiments can be slow on CPU. A free Colab GPU is
recommended for rerunning the full notebook. The saved Logistic Regression
bundle is already included, so users only need to install the application
dependencies to run the web app.

## Deployment

The deployed application uses the same [`streamlit_app.py`](C:/Users/LENOVO/OneDrive/Desktop/machine%20learning/summative_NLP/streamlit_app.py)
entry point as local development. To deploy a new copy with Streamlit
Community Cloud:

1. Connect the GitHub repository
   `Umukundwa004/summative_NLP`.
2. Select the `main` branch.
3. Set **Main file path** to `streamlit_app.py`.
4. Use `requirements.txt` as the dependency file.
5. Confirm that `models/kinnews/` and its model files are present in the
   deployed repository.

Do not use `api/main.py` as the Streamlit main file; it is the separate
FastAPI entry point. For a separately hosted API, set `ALLOWED_ORIGINS` to
the permitted browser origins and optionally set `MODELS_DIR` to a custom
model directory.

## License and data notice
 The KINNEWS
dataset contains news articles whose copyright belongs to their original
sources. Follow the dataset's terms and source notices when downloading,
redistributing, or publishing derived data.
