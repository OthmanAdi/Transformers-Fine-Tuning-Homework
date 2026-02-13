# Transformers Fine-Tuning Homework

## KI & Python Modul - Woche 2
**Dozent:** Ahmad Othman Adi | **Morphos GmbH**

Fine-Tuning von Transformer-Modellen auf echten Produktionsdaten mit Hugging Face.

---

## Was dieses Repo enthaelt

| Datei | Beschreibung | Level |
|-------|-------------|-------|
| `01_datasets_erkunden.py` | Echte Datasets von Hugging Face Hub laden | Pflicht |
| `02_finetuning_imdb.py` | Fine-Tuning auf 1.000 echten IMDB Reviews | Pflicht |
| `03_eigenes_dataset.py` | Eigenes deutsches Dataset + deutsches Model | Pflicht |
| `04_verstaendnis.py` | 6 Verstaendnis-Fragen | Pflicht |
| `05_bronze_model_vergleich.py` | 4 Models im Vergleich | Bronze |
| `06_silver_german_finetuning.py` | Deutsches Fine-Tuning auf Amazon Reviews | Silver |
| `07_gold_vit.py` | Vision Transformer auf CIFAR-10 | Gold |
| `08_diamond_pipeline.py` | Komplette Pipeline mit Dashboard | Diamond |

## Setup

```bash
pip install transformers datasets evaluate accelerate scikit-learn matplotlib torch
```

## Datasets

| Dataset | Sprache | Labels | Groesse | Code |
|---------|---------|--------|---------|------|
| IMDB | EN | 2 (pos/neg) | 50K | `load_dataset("stanfordnlp/imdb")` |
| Amazon DE | DE | 5 (Sterne) | 200K | `load_dataset("mteb/amazon_reviews_multi", "de")` |
| Multilingual | 12 Sprachen | 3 | 591K | `load_dataset("tyqiangz/multilingual-sentiments", "german")` |
| CIFAR-10 | - | 10 Klassen | 60K Bilder | `load_dataset("cifar10")` |

## Models

| Model | Sprache | Parameter | ID |
|-------|---------|-----------|-----|
| DistilBERT | EN | 67M | `distilbert-base-uncased` |
| BERT German | DE | 110M | `dbmdz/bert-base-german-cased` |
| GBERT | DE | 110M | `deepset/gbert-base` |
| GELECTRA | DE | 110M | `deepset/gelectra-base` |
| German Sentiment | DE | 110M | `oliverguhr/german-sentiment-bert` |
| ViT Base | - | 86M | `google/vit-base-patch16-224` |

## Abgabe

Vor der naechsten Vorlesung. Pflicht + mindestens 1 Challenge.

---

*Morphos GmbH - KI & Python Modul*

