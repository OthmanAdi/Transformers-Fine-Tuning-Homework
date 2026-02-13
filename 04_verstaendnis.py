# ============================================
# VERSTAENDNIS-FRAGEN - PFLICHT
# ============================================
# Beantwortet JEDE Frage in eigenen Worten.
# Copy-Paste von ChatGPT = 0 Punkte.
# Schreibt die Antworten als Kommentare hier rein.

# FRAGE 1: learning_rate
# In der Vorlesung war learning_rate=2e-5 (also 0.00002).
# Euer CNN von Woche 1 hatte learning_rate=0.001.
# a) Warum ist die Learning Rate beim Fine-Tuning 50x kleiner?
# b) Was passiert wenn ihr learning_rate=0.1 benutzt?
# c) Wie heisst das Problem wenn die LR zu gross ist? (Englischer Fachbegriff)
# Eure Antwort: ___

# FRAGE 2: Datasets
# In Aufgabe 1.2 habt ihr load_dataset("stanfordnlp/imdb") benutzt.
# In Aufgabe 1.3 habt ihr eigene Listen geschrieben.
# a) Was ist der Vorteil von load_dataset() gegenueber eigenen Listen?
# b) Warum haben wir nur 500 Reviews genommen statt alle 25.000?
# c) Was wuerde passieren wenn wir ALLE 25.000 auf CPU nehmen?
# Eure Antwort: ___

# FRAGE 3: Metriken
# In der Vorlesung hatten wir nur eval_loss.
# In Aufgabe 1.2 habt ihr compute_metrics mit Accuracy und F1 benutzt.
# a) Warum reicht Accuracy NICHT aus? Gebt ein konkretes Beispiel.
# b) Was ist der Unterschied zwischen Precision und Recall?
# c) Was zeigt die Confusion Matrix was Accuracy alleine nicht zeigt?
# Eure Antwort: ___

# FRAGE 4: Deutsche vs Englische Models
# a) Warum funktioniert "distilbert-base-uncased" schlecht auf deutschen Texten?
# b) Was bedeutet "cased" vs "uncased"?
# c) Welches deutsche Model habt ihr gewaehlt und warum?
# Eure Antwort: ___

# FRAGE 5: Fine-Tuning vs Transfer Learning
# Letzte Woche: VGG16 auf Intel-Bilder (Transfer Learning)
# Diese Woche: DistilBERT auf IMDB (Fine-Tuning)
# a) Ist Fine-Tuning das gleiche wie Transfer Learning? Erklaert.
# b) Bei VGG16 haben wir Layers eingefroren (base_model.trainable=False).
#    Beim DistilBERT waren ALLE 66M Parameter trainable.
#    Was ist der Unterschied in der Strategie?
# c) Wann friert man Layers ein, wann nicht?
# Eure Antwort: ___

# FRAGE 6: DataCollator vs padding=True
# In der Vorlesung: tokenizer(texts, padding=True) = alle Texte auf gleiche Laenge
# In Aufgabe 1.2: DataCollatorWithPadding = Padding pro Batch
# a) Was ist der Unterschied?
# b) Warum ist DataCollator effizienter?
# c) Gebt ein Beispiel mit konkreten Zahlen.
#    (z.B. 3 Texte mit Laenge 10, 50, 200)
# Eure Antwort: ___
