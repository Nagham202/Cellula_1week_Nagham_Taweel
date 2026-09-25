import json
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

pd.set_option("display.max_colwidth", 70)
pd.set_option("display.width", 200)

DATA_PATH = Path(__file__).resolve().parent.parent / "toxic_data.csv"

CLASSES_TO_DROP = ["Elections", "Sex-Related Crimes",
                   "Child Sexual Exploitation", "Suicide & Self-Harm"]

SEED = 42

USE_IMAGE_DESCRIPTION = True

MAX_LEN = 60
VOCAB_SIZE = None

EMBEDDING_DIM = 64
LSTM_UNITS = 64
DROPOUT = 0.3
BIDIRECTIONAL = False
LEARNING_RATE = 0.001

BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 3
USE_CLASS_WEIGHTS = True

RUN_TEST = True

OUTPUT_DIR = Path(__file__).resolve().parent

df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)

df = df.rename(columns={"query": "text", "image descriptions": "image", "Toxic Category": "label"})

df = df[~df["label"].isin(CLASSES_TO_DROP)]
print("After dropping the 4 tiny classes:", df.shape)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

if USE_IMAGE_DESCRIPTION:
    df["clean_text"] = df["clean_text"] + " xxsep " + df["image"].apply(clean_text)

print("\nBefore vs after cleaning:")
print(df[["text", "clean_text"]].sample(5, random_state=SEED))

lengths = df["clean_text"].str.split().str.len()
print("\nInput length (words): 99% are at most", int(lengths.quantile(0.99)), "| longest:", lengths.max())

labels_per_text = df.groupby("clean_text")["label"].nunique()
print("\nTexts that appear with more than one label:", (labels_per_text > 1).sum())

df = df.drop_duplicates(subset="clean_text")
print("After removing duplicate texts:", df.shape)

print("\nClass distribution:")
counts = df["label"].value_counts()
percents = (df["label"].value_counts(normalize=True) * 100).round(1)
print(pd.DataFrame({"count": counts, "percent": percents}))

class_names = sorted(df["label"].unique())
label_to_id = {name: i for i, name in enumerate(class_names)}
df["label_id"] = df["label"].map(label_to_id)
print("\nLabel -> number:", label_to_id)

X = df["clean_text"].values
y = df["label_id"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

print(f"\nTrain: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

split_counts = pd.DataFrame({
    "train": pd.Series(y_train).value_counts(),
    "val": pd.Series(y_val).value_counts(),
    "test": pd.Series(y_test).value_counts(),
}).sort_index()
split_counts.index = class_names
print(split_counts)

overlap = set(X_train) & (set(X_val) | set(X_test))
print("Texts in train that also appear in val/test:", len(overlap))

word_counts = Counter(word for sentence in X_train for word in sentence.split())
print("\nUnique words in train:", len(word_counts))
print("Words that appear only once:", sum(1 for c in word_counts.values() if c == 1))
print("10 most common:", word_counts.most_common(10))

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    standardize=None,
    split="whitespace",
    output_mode="int",
    output_sequence_length=MAX_LEN,
)
vectorizer.adapt(X_train)

vocab = vectorizer.get_vocabulary()
print("\nVocabulary size (including padding and [UNK]):", len(vocab))
print("First 12 entries:", [str(word) for word in vocab[:12]])

X_train_seq = vectorizer(X_train).numpy()
X_val_seq = vectorizer(X_val).numpy()
X_test_seq = vectorizer(X_test).numpy()

print("\nShape of X_train_seq:", X_train_seq.shape)
print("Example sentence:", X_train[0])
print("As IDs:", X_train_seq[0])

unknown = (X_val_seq == 1).sum()
real_words = (X_val_seq > 0).sum()
print(f"Unknown words in validation: {unknown / real_words:.1%}")

tf.keras.utils.set_random_seed(SEED)

lstm_layer = tf.keras.layers.LSTM(LSTM_UNITS)
if BIDIRECTIONAL:
    lstm_layer = tf.keras.layers.Bidirectional(lstm_layer)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(MAX_LEN,)),
    tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=EMBEDDING_DIM, mask_zero=True),
    lstm_layer,
    tf.keras.layers.Dropout(DROPOUT),
    tf.keras.layers.Dense(len(class_names), activation="softmax"),
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["accuracy"],
)

model.summary()

untrained_prediction = model.predict(X_val_seq[:1], verbose=0)
print("\nUntrained prediction for one sentence:", untrained_prediction.round(3))
print("True class:", class_names[y_val[0]])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

if USE_CLASS_WEIGHTS:
    weights = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=y_train)
    class_weight = dict(enumerate(weights))
    print("\nClass weights:", {class_names[i]: round(float(w), 2) for i, w in class_weight.items()})
else:
    class_weight = None

start_time = time.time()
history = model.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    class_weight=class_weight,
    verbose=2,
)
training_time = time.time() - start_time
epochs_run = len(history.history["loss"])
print(f"\nTraining took {training_time:.1f} s and ran {epochs_run} epochs")

val_probabilities = model.predict(X_val_seq, verbose=0)
val_predictions = val_probabilities.argmax(axis=1)

print("\nValidation macro F1:   ", round(f1_score(y_val, val_predictions, average="macro", zero_division=0), 3))
print("Validation weighted F1:", round(f1_score(y_val, val_predictions, average="weighted", zero_division=0), 3))
print("\nPer-class results on validation:")
print(classification_report(y_val, val_predictions, target_names=class_names, digits=3, zero_division=0))


TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"

epoch_numbers = range(1, epochs_run + 1)
best_epoch = int(np.argmin(history.history["val_loss"])) + 1

fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(11, 4))

loss_ax.plot(epoch_numbers, history.history["loss"], color=TRAIN_COLOR, linewidth=2, marker="o", label="Training")
loss_ax.plot(epoch_numbers, history.history["val_loss"], color=VAL_COLOR, linewidth=2, marker="o",
             linestyle="--", label="Validation")
loss_ax.set_title("Loss per epoch (lower is better)")
loss_ax.set_ylabel("Loss")

acc_ax.plot(epoch_numbers, history.history["accuracy"], color=TRAIN_COLOR, linewidth=2, marker="o", label="Training")
acc_ax.plot(epoch_numbers, history.history["val_accuracy"], color=VAL_COLOR, linewidth=2, marker="o",
            linestyle="--", label="Validation")
acc_ax.set_title("Accuracy per epoch (higher is better)")
acc_ax.set_ylabel("Accuracy")

for ax in (loss_ax, acc_ax):
    ax.axvline(best_epoch, color="gray", linewidth=1, linestyle=":")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
loss_ax.annotate(f"best epoch ({best_epoch})", xy=(best_epoch, loss_ax.get_ylim()[1]),
                 xytext=(4, -12), textcoords="offset points", color="gray", fontsize=9)

fig.suptitle("LSTM training history")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
plt.close(fig)
print("\nSaved", OUTPUT_DIR / "training_curves.png")

if RUN_TEST:
    test_predictions = model.predict(X_test_seq, verbose=0).argmax(axis=1)

    test_results = {
        "macro_f1": f1_score(y_test, test_predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, test_predictions, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_test, test_predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, test_predictions, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_test, test_predictions, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_test, test_predictions, average="weighted", zero_division=0),
    }
    print("\n===== TEST SET RESULTS =====")
    for name, value in test_results.items():
        print(f"{name:20s} {value:.3f}")
    print("\nPer-class results on test:")
    print(classification_report(y_test, test_predictions, target_names=class_names, digits=3, zero_division=0))

    matrix = confusion_matrix(y_test, test_predictions)

    blues = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "blues", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap=blues)
    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("LSTM confusion matrix (test set)")
    for row in range(len(class_names)):
        for col in range(len(class_names)):
            count = matrix[row, col]
            text_color = "white" if count > matrix.max() / 2 else "#0b0b0b"
            ax.text(col, row, count, ha="center", va="center", color=text_color, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("Saved", OUTPUT_DIR / "confusion_matrix.png")

    model.save(OUTPUT_DIR / "lstm_model.keras")
    print("Saved", OUTPUT_DIR / "lstm_model.keras")

    results = {
        "settings": {
            "use_image_description": USE_IMAGE_DESCRIPTION, "max_len": MAX_LEN, "vocab_size": len(vocab),
            "embedding_dim": EMBEDDING_DIM, "lstm_units": LSTM_UNITS, "dropout": DROPOUT, "bidirectional": BIDIRECTIONAL, "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE, "max_epochs": EPOCHS, "patience": PATIENCE,
            "use_class_weights": USE_CLASS_WEIGHTS, "seed": SEED,
        },
        "data": {"train": len(X_train), "val": len(X_val), "test": len(X_test), "classes": class_names},
        "training": {"epochs_run": epochs_run, "best_epoch": best_epoch, "seconds": round(training_time, 1),
                     "parameters": model.count_params()},
        "validation": {
            "macro_f1": f1_score(y_val, val_predictions, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_val, val_predictions, average="weighted", zero_division=0),
        },
        "test": test_results,
        "test_per_class": classification_report(y_test, test_predictions, target_names=class_names,
                                                zero_division=0, output_dict=True),
        "confusion_matrix": matrix.tolist(),
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("Saved", OUTPUT_DIR / "results.json")
