# 🔍 TruthLens AI – Fake News Detection System

A Machine Learning web app that checks whether a news article looks **REAL** or **FAKE**.
Built with Python, Scikit-learn (TF-IDF + 4 models) and Streamlit.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-app-red) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

<!-- Add screenshots here after running the app, e.g.:
![Detect page](screenshots/detect.png)
![Dashboard](screenshots/dashboard.png)
-->

## ✨ Features
- **News detection** by pasting text or entering a **URL**
- **4 ML models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Compare mode**: all 4 models predict and the final answer is a majority vote
- **Explainable AI**: shows the words that pushed the verdict towards FAKE or REAL
- **Red Flags**: ALL-CAPS, many `!`, clickbait phrases, missing sources
- **Trust Score (0–100)**: model confidence combined with red flags
- **Hindi support**: Hindi text is translated to English before analysis (lower accuracy)
- **Login system** with per-user history (SQLite, hashed passwords)
- **Dashboard** with prediction statistics and charts
- **History** page with filter and CSV download
- **User feedback** (👍 / 👎) and user-verified accuracy
- **Model Performance** page: accuracy, precision, recall, F1, confusion matrix

## 📊 Model Results
Dataset: [Fake and Real News Dataset (ISOT)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) – 38,631 articles after cleaning, 80/20 train-test split.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 98.01% | 98.01% | 98.01% | 98.01% |
| Random Forest | 97.72% | 97.73% | 97.72% | 97.72% |
| Gradient Boosting | 96.61% | 96.63% | 96.61% | 96.60% |
| Decision Tree | 94.64% | 94.65% | 94.64% | 94.64% |

The agency tag "(Reuters)" is removed from the text before training, so the models learn writing style instead of a shortcut word.

## 🧠 How it works
1. Title + text are cleaned (lowercase, remove links and symbols).
2. **TF-IDF** converts text into 5,000 numeric features.
3. The 4 models predict REAL or FAKE with a confidence score.
4. Red flags and model confidence are combined into a Trust Score.

## 🗂 Project structure
```
fake-news-detector/
├── app.py            # Streamlit UI
├── train_models.py   # trains and saves the 4 models
├── auth.py           # register / login
├── database.py       # SQLite history + feedback
├── utils.py          # cleaning, explain, red flags, URL, Hindi
├── requirements.txt
├── .streamlit/config.toml
├── data/             # put Fake.csv and True.csv here
└── models/           # trained models + metrics.json
```

## 🚀 Run locally (Windows / VS Code)
1. Download the dataset from Kaggle and put `Fake.csv` and `True.csv` in `data/`
   (only needed if you want to retrain; trained models are already in `models/`).
2. Open a terminal in the project folder:
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_models.py     # optional, takes about 3-4 minutes
streamlit run app.py
```
3. Open http://localhost:8501 and create an account.

If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## ⚠ Limitations
- The dataset is mostly US political news from 2016–2017, so results on other topics may be less reliable.
- The models learn writing style, **not facts**. They cannot verify whether a claim is true.
- Hindi news is machine-translated first, so accuracy is lower. URL and Hindi features need internet.
- Login is basic (no password reset or email verification) and meant for learning, not production.

Use this tool as a helper only and always cross-check with trusted sources.

## 🛠 Tech stack
Python · Pandas · Scikit-learn · Streamlit · Plotly · SQLite · BeautifulSoup · deep-translator

## 📄 License
Made for educational purposes. Dataset belongs to its original authors (see the Kaggle page).
