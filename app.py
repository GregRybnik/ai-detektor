# app.py
# pamiętaj aby w terminalu odpalić: /Users/macbookpro/Documents/skrypty/python3 app.py


from flask import Flask, render_template, request
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import re
from docx import Document
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model i tokenizer GPT2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Proste dzielenie na zdania
def simple_sent_tokenize(text):
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]

def calculate_perplexity(text, model, tokenizer, max_length=512):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = encodings.input_ids
    n_tokens = input_ids.shape[1]
    stride = max_length
    lls = []

    for i in range(0, n_tokens, stride):
        begin_loc = i
        end_loc = min(i + max_length, n_tokens)
        trg_len = end_loc - begin_loc
        input_ids_slice = input_ids[:, begin_loc:end_loc]
        with torch.no_grad():
            outputs = model(input_ids_slice, labels=input_ids_slice)
            neg_log_likelihood = outputs.loss * trg_len
        lls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / n_tokens)
    return ppl.item()

def calculate_burstiness(text):
    sentences = simple_sent_tokenize(text)
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    return np.std(sentence_lengths) if len(sentence_lengths) >= 2 else 0.0

@app.route('/', methods=['GET', 'POST'])
def index():
    wynik = None
    perplexity = None
    burstiness = None
    text = ""
    kolor = ""

    if request.method == 'POST':
        if request.form.get("clear") == "1":
            return render_template('index.html', wynik=None, kolor="", perplexity=None, burstiness=None, text="")

        uploaded_file = request.files.get('file')
        input_text = request.form.get('text')

        if uploaded_file and uploaded_file.filename.endswith('.docx'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
            uploaded_file.save(filepath)
            doc = Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif input_text:
            text = input_text.strip()

        if text:
            perplexity = calculate_perplexity(text, model, tokenizer)
            burstiness = calculate_burstiness(text)
            if perplexity < 60:
                wynik = "Tekst wygląda na wygenerowany przez AI."
                kolor = "red"
            elif 60 <= perplexity <= 100:
                wynik = "Tekst znajduje się na pograniczu – może być wygenerowany przez AI lub napisany przez człowieka."
                kolor = "orange"
            else:
                wynik = "Tekst wygląda na napisany przez człowieka. Cieszymy się ;-)"
                kolor = "green"

    return render_template('index.html', wynik=wynik, kolor=kolor, perplexity=perplexity, burstiness=burstiness, text=text)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

