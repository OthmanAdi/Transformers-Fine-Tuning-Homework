# ============================================
# AUFGABE 1.2: FINE-TUNING AUF ECHTEN DATEN
# ============================================
# Was ist anders als in der Vorlesung?
#
# 1. ECHTE Daten (500 IMDB Reviews statt 20 Saetze)
# 2. datasets Library statt manueller Listen
# 3. compute_metrics mit Accuracy + F1
# 4. Confusion Matrix am Ende
# 5. DataCollatorWithPadding (effizienter als padding=True)
#
# Installation:
# pip install transformers datasets evaluate accelerate scikit-learn matplotlib

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# ============================================
# SCHRITT 1: DATASET LADEN
# ============================================
# Wir nehmen nur 500 Reviews (nicht alle 25.000)
# Grund: CPU Training. In Produktion mit GPU nimmt man alles.

dataset = load_dataset("stanfordnlp/imdb")

small_train = dataset["train"].shuffle(seed=42).select(range(500))
small_test = dataset["test"].shuffle(seed=42).select(range(500))

print(f"Training:  {len(small_train)} Reviews")
print(f"Test:      {len(small_test)} Reviews")
print(f"Labels:    0=negativ, 1=positiv")

train_labels = small_train["label"]
print(f"  Positiv: {sum(train_labels)}")
print(f"  Negativ: {len(train_labels) - sum(train_labels)}")

# ============================================
# SCHRITT 2: TOKENIZER + TOKENISIEREN
# ============================================

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenisieren mit .map() - VIEL schneller als ReviewDataset Klasse
# WICHTIG: Kein padding hier! DataCollator macht das spaeter pro Batch.
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

tokenized_train = small_train.map(tokenize_function, batched=True)
tokenized_test = small_test.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print(f"\nTokenisiert. Bereit fuer Training.")

# ============================================
# SCHRITT 3: METRIKEN DEFINIEREN
# ============================================
# F1-Score = Kombination aus Precision und Recall
# Accuracy alleine luegt bei unbalancierten Daten!

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary"),
    }

# ============================================
# SCHRITT 4: MODEL LADEN
# ============================================

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {model_name}")
print(f"Parameter: {total_params:,}")

# ============================================
# SCHRITT 5: TRAINING
# ============================================

training_args = TrainingArguments(
    output_dir="./imdb_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("\n" + "=" * 60)
print("FINE-TUNING STARTET")
print("=" * 60)

trainer.train()

# ============================================
# SCHRITT 6: CLASSIFICATION REPORT
# ============================================

print("\n" + "=" * 60)
print("EVALUATION AUF TEST-SET")
print("=" * 60)

predictions = trainer.predict(tokenized_test)
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["NEGATIVE", "POSITIVE"]))

# ============================================
# SCHRITT 7: CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NEGATIVE", "POSITIVE"])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("IMDB Fine-Tuning - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_imdb.png", dpi=150)
print("\nConfusion Matrix gespeichert: confusion_matrix_imdb.png")
plt.show()

# ============================================
# SCHRITT 8: SPEICHERN + TESTEN
# ============================================

save_path = "./mein_imdb_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\nModel gespeichert: {save_path}/")

from transformers import pipeline

mein_classifier = pipeline("text-classification", model=save_path, tokenizer=save_path)

test_saetze = [
    "This movie was absolutely incredible, best I've seen all year!",
    "Terrible waste of time, awful acting and boring plot.",
    "It was okay, nothing special but not terrible either.",
    "A masterpiece of cinema, truly groundbreaking filmmaking.",
    "I fell asleep halfway through, so predictable and dull.",
]

print("\n" + "=" * 60)
print("EUER MODEL - LIVE TEST")
print("=" * 60)

for satz in test_saetze:
    result = mein_classifier(satz)[0]
    bar = "█" * int(result["score"] * 20)
    print(f"  {result['label']:10s} [{bar:20s}] {result['score']:.1%}")
    print(f"  Text: {satz}\n")
