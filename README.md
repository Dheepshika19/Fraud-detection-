# Fake News Detection

This project uses machine learning to classify news articles as real or fake. It leverages a logistic regression model trained on TF-IDF features extracted from news text.

## Project Structure

- `model_training.py` — Script to train the fake news classifier and save the model/vectorizer.
- `predict_fake_news.py` — (Not shown here) Script for making predictions using the trained model.
- `Fake.csv` — Dataset of fake news articles.
- `True.csv` — Dataset of real news articles.
- `fake_news_model.pkl` — Saved logistic regression model.
- `tfidf_vectorizer.pkl` — Saved TF-IDF vectorizer.

## Model Training

The model is trained using the following steps (see [`model_training.py`](model_training.py)):

1. Load and label the datasets (`Fake.csv` as 0, `True.csv` as 1).
2. Combine and shuffle the data.
3. Split into training and test sets.
4. Vectorize the text using TF-IDF.
5. Train a logistic regression classifier.
6. Evaluate accuracy on the test set.
7. Save the trained model and vectorizer.

## Usage

### Train the Model

```sh
python model_training.py
```

### Predict New Articles

Use the `predict_fake_news.py` script (ensure `fake_news_model.pkl` and `tfidf_vectorizer.pkl` are present).

## Requirements

- Python 3.x
- pandas
- scikit-learn
- joblib

Install dependencies:

```sh
pip install pandas scikit-learn joblib
```# Fake News Detection App

A machine learning project to classify news articles as **real** or **fake** using logistic regression and TF-IDF features.

---

## Features

- **Train** a logistic regression model on labeled news datasets.
- **Predict** if a news article is real or fake using the trained model.
- **Save** and **reuse** the model and vectorizer for fast predictions.

---

## Project Structure

```
fake news dection app/
│
├── model_training.py         # Script to train and save the model/vectorizer
├── predict_fake_news.py      # Script to predict news authenticity
├── Fake.csv                  # Dataset: fake news articles
├── True.csv                  # Dataset: real news articles
├── fake_news_model.pkl       # Saved logistic regression model (generated)
├── tfidf_vectorizer.pkl      # Saved TF-IDF vectorizer (generated)
```

---

## How It Works

### 1. Training (`model_training.py`)

- Loads `Fake.csv` and `True.csv`, labeling fake as `0` and real as `1`.
- Combines, shuffles, and splits the data into training and test sets.
- Vectorizes the text using TF-IDF.
- Trains a logistic regression classifier.
- Evaluates and prints the accuracy.
- Saves the trained model and vectorizer as `.pkl` files.

**Run training:**
```sh
python model_training.py
```

---

### 2. Prediction (`predict_fake_news.py`)

- Loads the saved model and vectorizer.
- Prompts the user to enter a news article or paragraph.
- Transforms the input and predicts its authenticity.
- Prints whether the news is **REAL** or **FAKE**.

**Run prediction:**
```sh
python predict_fake_news.py
```

---

## Example Usage

**Training Output:**
```
✅ Script started!
📥 Loading data...
✅ Data prepared!
⚙️  Vectorizing...
🚀 Training model...
📊 Evaluating...
✅ Accuracy: 0.98
💾 Saving model...
🎉 Model training complete and saved.
```

**Prediction Output:**
```
📰 Enter a news article or paragraph:
<your news text here>
✅ This news is REAL.
```

---

## Requirements

- Python 3.x
- pandas
- scikit-learn
- joblib

**Install dependencies:**
```sh
pip install pandas scikit-learn joblib
```

---

## Notes

- Ensure `Fake.csv` and `True.csv` are present in the project directory before training.
- The model and vectorizer (`.pkl` files) are generated after running `model_training.py`.

---

## License

MIT License

---

## Author

Karneka C
