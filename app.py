# app.py
# pamiętaj aby w terminalu odpalić: /Users/macbookpro/Documents/skrypty/python3 app.py
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
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
app.secret_key = "supersekretnyklucz"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)

# Model użytkownika
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Tworzenie konta admina przy pierwszym uruchomieniu
@app.before_first_request
def create_admin():
    db.create_all()
    if not User.query.filter_by(username="admin").first():
        admin = User(username="admin", password=generate_password_hash("1234567890AaA"))
        db.session.add(admin)
        db.session.commit()
        print("✅ Konto 'admin' zostało utworzone.")

# Model językowy
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Tokenizacja zdań
def simple_sent_tokenize(text):
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]

# Perplexity
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

# Burstiness
def calculate_burstiness(text):
    sentences = simple_sent_tokenize(text)
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    return np.std(sentence_lengths) if len(sentence_lengths) >= 2 else 0.0

# Rejestracja – tylko dla admina
@app.route('/register', methods=['GET', 'POST'])
def register():
    if not session.get('logged_in') or session.get('user') != 'admin':
        return redirect(url_for('login'))

    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        if User.query.filter_by(username=username).first():
            return "Użytkownik już istnieje."
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('users'))
    return render_template('register.html')

# Lista użytkowników – tylko dla admina
@app.route('/users')
def users():
    if not session.get('logged_in') or session.get('user') != 'admin':
        return redirect(url_for('login'))
    all_users = User.query.all()
    return render_template('users.html', users=all_users)

# Usuwanie użytkownika – tylko dla admina
@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if not session.get('logged_in') or session.get('user') != 'admin':
        return redirect(url_for('login'))

    user = User.query.get(user_id)
    if user and user.username != 'admin':
        db.session.delete(user)
        db.session.commit()
    return redirect(url_for('users'))

# Logowanie
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            session['logged_in'] = True
            session['user'] = user.username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Błędna nazwa użytkownika lub hasło')
    return render_template('login.html')

# Wylogowanie
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Strona główna – analiza tekstu
@app.route('/', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

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
                wynik = "Tekst znajduje się na pograniczu AI / człowiek."
                kolor = "orange"
            else:
                wynik = "Tekst wygląda na napisany przez człowieka."
                kolor = "green"

    return render_template('index.html', wynik=wynik, kolor=kolor, perplexity=perplexity, burstiness=burstiness, text=text)

# Uruchomienie aplikacji
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
