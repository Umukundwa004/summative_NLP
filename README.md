# Kinyarwanda News Topic Classifier

Classifies a Kinyarwanda news article into one of 14 topics.

- **Live app:** https://summativenlp-mvcz2sv7jk56kouu45pdmg.streamlit.app/
- **Demo video:** https://youtu.be/AlPnrd_C270
- **GitHub:** https://github.com/Umukundwa004/summative_NLP

## Problem

Automatic topic classification of Kinyarwanda news helps organise archives,
match articles to readers, and monitor public information in the language most
Rwandans speak. It is hard because Kinyarwanda is a Bantu language with rich
morphology (one word appears in many forms) and, like most African languages,
has little labelled data and few ready made NLP tools.

Topics: `politics`, `sport`, `economy`, `health`, `entertainment`, `history`,
`technology`, `tourism`, `culture`, `fashion`, `religion`, `environment`,
`education`, `relationship`.

## Dataset

**KINNEWS**: real Kinyarwanda news articles from Rwandan websites and
newspapers ([Hugging Face](https://huggingface.co/datasets/andreniyongabo/kinnews_kirnews)).

- 21,268 raw articles. About 48% were duplicates, and 52% of the official test
  articles also appeared in the training file.
- After cleaning (1,453 conflicting-label and 8,891 duplicate articles removed):
  **10,924 articles**, split 70/15/15 (train/validation/test) with
  stratification. The test set has 1,639 articles.
- Cleaning: unescape HTML, normalise Unicode, remove URLs and emails, lower-case.

## Models and results

All models use the same split. The best model is chosen on validation macro-F1.

| Model | Test macro-F1 | Test accuracy |
|---|---:|---:|
| **TF-IDF + Logistic Regression** | **0.7125** | 0.7694 |
| Ensemble of 6 models | 0.7024 | 0.7730 |
| BiLSTM + Word2Vec | 0.6667 | 0.7492 |
| TF-IDF + Complement Naive Bayes | 0.6479 | **0.7779** |
| TextCNN | 0.6151 | 0.6608 |
| AfroXLMR (trained on 800 articles) | 0.5608 | 0.6827 |
| BiLSTM + attention | 0.4637 | 0.5265 |
| Majority-class baseline | 0.0229 | 0.1910 |

**Deployed model:** TF-IDF (word unigrams and bigrams, 300,000 features) with
balanced Logistic Regression. It also had the best validation macro-F1 (0.7074)
and 5-fold cross-validation macro-F1 (0.7252).

What we learned:
- A simple model beat the neural networks on this small dataset.
- Word2Vec embeddings helped the BiLSTM a lot (validation macro-F1 0.482 to 0.679).
- AfroXLMR used only about 10% of the training data (compute limits), so it is a
  small test, not a fair comparison.

## Error analysis

The deployed model got 378 of 1,639 test articles wrong (23.1%).

- **Works well:** sport (F1 0.94), religion (0.84), entertainment (0.82), relationship (0.82).
- **Works poorly:** education, culture, environment, history (F1 0.56 to 0.60).
  These topics have few training articles and no fixed vocabulary.
- **Common confusions:** politics, economy, and health overlap. 
- **Other causes:** noisy labels, and Kinyarwanda words that were  not seen in training.

## Limitations

- Small, imbalanced data; some topics have under 200 training articles.
- Only exact duplicates were removed, and labels were not manually checked.
- Each article has one label, but real news can belong to several topics.
- Trained on news only; it may do worse on social media or code-mixed text.

**Future work:** fine-tune AfroXLMR on all the training data, add character
n-gram features, review labels with native speakers, and show a low-confidence
warning in the app.

## Project structure

```text
api/                 FastAPI server, shared prediction code, tests
data/KINNEWS/        Dataset files (not in Git)
frontend/index.html  Optional HTML client for the API
models/kinnews/      Saved Logistic Regression model
notebook/            Training, experiments, evaluation
results/kinnews/     Metrics and error-analysis outputs
streamlit_app.py     Web app
requirements.txt     Dependencies
```

## Run locally

Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open `http://localhost:8501`, paste Kinyarwanda text, and get the predicted
topic, confidence, and class probabilities.

**Optional API:**

```bash
python -m pip install -r api/requirements.txt
uvicorn main:app --app-dir api --port 8000
```

Endpoints: `GET /health`, `POST /predict`, `GET /docs`. Run tests with
`pytest api -q`.

## Reproduce the experiments

1. Open `notebook/Kinnews_news_classification.ipynb` in Jupyter or Google Colab
   (a GPU is recommended; AfroXLMR needs one).
2. Set `DATASET = "kinnews"` and run the cells in order.
3. The notebook cleans and splits the data, trains every model, evaluates them,
   and saves the results to `results/kinnews/` and the best model to
   `models/kinnews/`.

## Deployment

Deployed with Streamlit Community Cloud from this repository (`main` branch,
main file `streamlit_app.py`, dependencies in `requirements.txt`). The app
loads the saved model from `models/kinnews/` at startup.

## Existing resources vs. my own work

**Existing:** the KINNEWS dataset, the pretrained AfroXLMR model
(`Davlan/afro-xlmr-base`), and the libraries scikit-learn, PyTorch, gensim,
Hugging Face Transformers, FastAPI, and Streamlit.

**My own work:** data analysis, cleaning and the shared  data split, training and
tuning of all models and the ensemble, experiments evaluation, and error
analysis, the saved model, web app, API, and tests. AI tools (Copilot, ChatGPT)
were used for code fixes and editing.

## License and data notice

For coursework and experimentation. KINNEWS articles belong to their original
sources; follow the dataset's terms when sharing or publishing derived data.
