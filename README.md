# Week 1 — Toxic Content Classification with RNN and LSTM

Part of my NLP internship at **Cellula Technologies** (Week 1).

## What it does
Classifies a user query (together with a description of an attached image) into one of five
safety categories: **Safe, Violent Crimes, Non-Violent Crimes, unsafe, Unknown S-Type**.
I built and compared two deep learning models for this: a simple **RNN** and an **LSTM**.

## Tech stack
Python, TensorFlow / Keras, pandas, scikit-learn, matplotlib

## How it works
1. **Data exploration** (`explore_data.py`): shape, missing values, duplicates, class balance.
2. **Cleaning**: lowercase, remove punctuation, drop duplicate rows, and drop 4 classes
   that had only 2–5 unique examples.
3. **Input**: the query and the image description are joined with a separator token (`xxsep`),
   because the label depends on both.
4. **Model**: TextVectorization → Embedding (64) → RNN / LSTM (64 units) → Dense softmax.
5. **Training**: class weights for the imbalanced classes, early stopping (patience 3),
   train / validation / test split of 1407 / 301 / 302.
6. **Evaluation**: macro and weighted F1, per-class report, confusion matrix, training curves.

## Results (test set)
| Model | Parameters | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| RNN  | 247,877 | 86.8% | 0.784 | 0.888 |
| LSTM | 272,645 | **92.7%** | **0.814** | **0.927** |

The LSTM beat the simple RNN on every metric. Both models struggle with the small
"Unknown S-Type" class (13 test examples, F1 ≈ 0.15), which is often confused with "Safe".

## Project structure
```
RNN/   train_rnn.py, make_report.py, results.json, rnn_model.keras, plots, RNN_report.pdf
LSTM/  train_lstm.py, make_report.py, results.json, lstm_model.keras, plots, LSTM_report.pdf
explore_data.py, toxic_data.csv, requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
python LSTM/train_lstm.py
```
