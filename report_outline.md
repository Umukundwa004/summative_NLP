# Report outline: Text Classification for an African Language
**Kinyarwanda news topic classification (KINNEWS), served by FastAPI with an HTML page**

Parts marked **[FILL IN]** need your own results. They come from the notebook (`results/<dataset>/all_results.txt` and the CSV files) after you run it. Everything else describes what the project already does.

---

## 1. Introduction and problem
- **Problem:** assign a news article written in Kinyarwanda to a topic (politics, sport, economy, health, ...).
- **Why it matters:** news organisations and readers need topic tags for search, recommendation and archiving. Kinyarwanda is a low-resource language with few NLP tools.
- **Language used:** Kinyarwanda (`kinnews`).

## 2. Data
### 2.1 How the data was obtained
- **KINNEWS** (Kinyarwanda), Niyongabo et al., *KINNEWS: Benchmarking Text Classification for Kinyarwanda*, COLING 2020 (arXiv 2010.12174).
- KINNEWS contains 21,268 articles from 20 Rwandan news sources (15 websites and 5 newspapers), covering 14 topics.
- Public KINNEWS dataset hosted on Hugging Face; the files are downloaded into the project's `data/` folder (Section 2).
- Each article has a title, content and a topic label supplied with the dataset.
- **[FILL IN]** licence and ethics: the readme states that the copyright of the articles belongs to the original news sources; the Hugging Face page lists the dataset script under the MIT licence. State how you use the data (study purposes).

### 2.2 How the categories were defined
- The 14 topics are numbered in the files: politics, sport, economy, health, entertainment, history, technology, tourism, culture, fashion, religion, environment, education, and relationship.
- **[FILL IN]** the class table (the notebook prints it) and the **imbalance** (for example politics vs. education).
- **[FILL IN]** which topics were removed for having fewer than `MIN_ARTICLES_PER_CLASS` unique articles, and why (they cannot be learned or fairly tested).

### 2.3 Data quality problems found (important!)
- A large share of the articles are **duplicates**, and some **test articles also appear in the train file**. Without removing them, test scores would be inflated. **[FILL IN]** the numbers printed in Section 3 of the notebook.
- Some identical texts have **two different topic labels**. **[FILL IN]** how many (Section 4). These were removed because the correct label cannot be known.
- After cleaning, **[FILL IN]** how many articles remain (as a percentage of the original).

### 2.4 Preprocessing (Section 4, function `basic_cleaning`)
1. decode HTML codes
2. unicode normalisation (NFKC)
3. remove invisible characters
4. remove links and e-mail addresses
5. lower-case
6. squeeze spaces

Then: remove empty articles, articles with conflicting labels, and duplicates. Deliberate choices: no stop-word removal or stemming (no reliable tools for these languages), numbers and punctuation kept.

### 2.5 Train / validation / test split (Section 5)
- All cleaned unique articles are split once into **70 % train / 15 % validation / 15 % test**, stratified by topic, fixed seed (42).
- The same three sets are used by **every** experiment. Train: fitting; validation: choosing models and early stopping; **test: used once**.
- **[FILL IN]** the split table (articles per topic in train / validation / test).

## 3. Methods
| # | Model | Course topic |
|---|---|---|
| 0 | Majority-class baseline | baseline |
| 1 | TF-IDF + Logistic Regression (word 1-2 grams, balanced class weights) | classical baseline |
| 2 | TF-IDF + Complement Naive Bayes | classical baseline for imbalance |
| 3 | 1D CNN | neural text classification |
| 4 | BiLSTM + attention | RNN / LSTM |
| 5 | BiLSTM initialised with Word2Vec | word embeddings |
| 6 | Fine-tuned Transformer (AfroXLMR, which was pretrained on Kinyarwanda) | Transformer / contextual embeddings |
| 7 | Ensemble (average of probabilities) | combination |

Explain each in 2-4 sentences (all settings are in the notebook's `settings` dictionaries). State the **baseline** clearly (Experiments 0 and 1), and say which experiments you could not run (for example the Transformer if you had no GPU).

## 4. Evaluation
- **Metrics:** accuracy, precision, recall and F1 per topic, **macro-F1** (main metric, because the topics are very imbalanced), weighted-F1, log loss.
- **[FILL IN]** results table: all models on validation (Section 7) and on the test set (Section 8), plus 5-fold cross-validation for the classical models.
- **[FILL IN]** per-topic precision / recall / F1 of the best model and its confusion matrix; which topics are strong and weak.
- Compare every model with the baseline and explain why the best model wins.
- Note that small topics have very few test articles, so their scores are noisy.

## 5. Error analysis (notebook Section 9)
Use `results/<dataset>/misclassified_test.csv` and the printed tables. Cover:
1. which topics are confused (top confusions)
2. error rate per topic versus the number of training articles
3. error rate by article length (the networks read only the first `MAX_LEN` words)
4. confidence when wrong
5. **[FILL IN]** 3-6 example mistakes with your explanation. Reasons to consider: overlapping topics, rare topics, long articles truncated, noisy or conflicting labels, differences between news sources, rich word forms in Kinyarwanda that become `<unk>`, and the effect of removing duplicates.

Link every reason to evidence (a number or an example).

## 6. Deployment
- **Model hand-over:** the notebook (Section 10) saves the chosen model into the **`models/<dataset>/`** folder and writes `api/nlp_utils.py` from the notebook's own functions, so cleaning and prediction are identical in the server. It also checks that the server's model gives the **same probabilities** as the notebook on 100 test articles.
- **Backend: FastAPI** (`api/main.py`). It loads the Kinyarwanda model in `models/` when it starts. Endpoints `GET /health`, `GET /languages`, and `POST /predict` (JSON with `text` and optional `language`). Automatic documentation is available at `/docs`. Input validation (empty or too long text) and clear errors (404, 422, 503) are included.
- **Front end: HTML** (`frontend/index.html`, one file, no external libraries), served by FastAPI at `/`. The user types or pastes Kinyarwanda text, clicks **Classify**, and sees the predicted topic, confidence, probability bars, and a note when the model is unsure.
- **Tests:** `pytest api` tests the API and the page delivery. **[FILL IN]** mention that you also tried the page in a browser.
- **[FILL IN]** screenshots of the page, and the URL if you deploy the server online.

## 7. Limitations, ethics and future work
- The deduplicated dataset is much smaller than the original (especially for small topics), and test sets for rare topics are tiny.
- Models learned from long news articles may be unsure on very short texts (for example one sentence).
- Labels can be noisy (articles with two different labels were found).
- Future work: more data for rare topics, multi-label classification (an article can belong to several topics), larger pretrained models, and improved transfer learning for Kinyarwanda.

## 8. Reproducibility
- Fixed seed, the same split for all models, every experiment's settings saved in `results/<dataset>/experiment_log.csv`.
- Library versions are written to `api/requirements.txt` by the notebook.
- **[FILL IN]** the link to your repository.
