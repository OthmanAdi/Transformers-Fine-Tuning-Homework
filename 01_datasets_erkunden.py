# ============================================
# AUFGABE 1.1: ECHTE DATASETS LADEN
# ============================================
# Die 'datasets' Library ist wie pip fuer Daten.
# Statt Daten selbst zu schreiben, ladet ihr echte
# Produktionsdaten mit einer Zeile Code.
#
# Installation: pip install datasets

from datasets import load_dataset

# ============================================
# Dataset 1: IMDB Film-Reviews (Englisch)
# 25.000 Training + 25.000 Test Reviews
# Labels: 0 = negativ, 1 = positiv
# ============================================
print("=" * 60)
print("DATASET 1: IMDB Movie Reviews")
print("=" * 60)

imdb = load_dataset("stanfordnlp/imdb")

print(f"  Splits:    {list(imdb.keys())}")
print(f"  Training:  {len(imdb['train'])} Reviews")
print(f"  Test:      {len(imdb['test'])} Reviews")
print(f"  Spalten:   {imdb['train'].column_names}")
print(f"  Beispiel:  {imdb['train'][0]['text'][:100]}...")
print(f"  Label:     {imdb['train'][0]['label']}")

# ============================================
# Dataset 2: Amazon Reviews (Deutsch!)
# 200.000 Training + 5.000 Test Reviews pro Sprache
# Labels: 0-4 (Sterne 1-5)
# ============================================
print("\n" + "=" * 60)
print("DATASET 2: Amazon Reviews (DEUTSCH)")
print("=" * 60)

amazon_de = load_dataset("mteb/amazon_reviews_multi", "de")

print(f"  Splits:    {list(amazon_de.keys())}")
print(f"  Training:  {len(amazon_de['train'])} Reviews")
print(f"  Test:      {len(amazon_de['test'])} Reviews")
print(f"  Spalten:   {amazon_de['train'].column_names}")
print(f"  Beispiel:  {amazon_de['train'][0]['text'][:100]}...")
print(f"  Label:     {amazon_de['train'][0]['label']}")

# ============================================
# Dataset 3: Multilingual Sentiments (12 Sprachen!)
# ============================================
print("\n" + "=" * 60)
print("DATASET 3: Multilingual Sentiments (Deutsch)")
print("=" * 60)

multi = load_dataset("tyqiangz/multilingual-sentiments", "german")

print(f"  Splits:    {list(multi.keys())}")
print(f"  Training:  {len(multi['train'])} Reviews")
print(f"  Spalten:   {multi['train'].column_names}")
print(f"  Beispiel:  {multi['train'][0]['text'][:100]}...")
print(f"  Label:     {multi['train'][0]['label']}")

# ============================================
# VERSTAENDNIS-FRAGEN (als Kommentare beantworten)
# ============================================

# FRAGE 1: Wie viele Sterne-Bewertungen hat das Amazon-Dataset?
#           Was ist der Unterschied zu IMDB (nur 2 Labels)?
# Eure Antwort: ___

# FRAGE 2: Warum ist load_dataset() besser als Daten selbst zu schreiben?
#           Nennt mindestens 3 Gruende.
# Eure Antwort: ___
