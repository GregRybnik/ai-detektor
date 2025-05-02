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
    tokens = db.Column(db.Integer, default=10)

# Tworzenie konta admina
def create_admin():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username="admin").first():
            admin = User(username="admin", password=generate_password_hash("1234567890AaA"), tokens=9999)
            db.session.add(admin)
            db.session.commit()
            print("✅ Konto 'admin' zostało utworzone.")

# Wczytanie modeli
models = {
    "distilgpt2": {
        "tokenizer": GPT2Tokenizer.from_pretrained("distilgpt2"),
        "model": GPT2LMHeadModel.from_pretrained("distilgpt2")
    },
    "gpt2": {
        "tokenizer": GPT2Tokenizer.from_pretrained("gpt2"),
        "model": GPT2LMHeadModel.from_pretrained("gpt2")
    }
}
for m in models.values():
    m["model"].eval()

# Funkcje pomocnicze
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

# Rejestracja
@app.route('/register', methods=['GET', 'POST'])
def register():
    if not session.get('logged_in') or session.get('user') != 'admin':
        return redirect(url_for('login'))

    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        if User.query.filter_by(username=username).first():
            return "Użytkownik już istnieje."
        new_user = User(username=username, password=password, tokens=10)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('users'))
    return render_template('register.html')

# Lista użytkowników
@app.route('/users')
def users():
    if not session.get('logged_in') or session.get('user') != 'admin':
        return redirect(url_for('login'))
    all_users = User.query.all()
    return render_template('users.html', users=all_users)

# Zmiana liczby tokenów
@app.route('/update_tokens/<int:user_id>', methods=['POST'])
def update_tokens(user_id):
    if not session.get('logged_in') or session.get('user') != 'admin':
        return redirect(url_for('login'))

    user = User.query.get(user_id)
    if user and user.username != 'admin':
        try:
            new_token_count = int(request.form['tokens'])
            user.tokens = new_token_count
            db.session.commit()
        except ValueError:
            pass
    return redirect(url_for('users'))

# Usuwanie użytkownika
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

# Strona główna – analiza
@app.route('/', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    current_user = User.query.filter_by(username=session.get('user')).first()
    wynik = None
    perplexity = None
    burstiness = None
    text = ""
    kolor = ""
    selected_model_name = "distilgpt2"

    if request.method == 'POST':
        selected_model_name = request.form.get("model", "distilgpt2")

        if request.form.get("clear") == "1":
            return render_template('index.html', wynik=None, kolor="", perplexity=None,
                                   burstiness=None, text="", model_name=selected_model_name,
                                   tokens=current_user.tokens)

        if current_user.tokens <= 0:
            wynik = "Brak tokenów – nie możesz już analizować tekstów."
            kolor = "gray"
            return render_template('index.html', wynik=wynik, kolor=kolor,
                                   perplexity=None, burstiness=None, text="",
                                   model_name=selected_model_name, tokens=current_user.tokens)

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
            tokenizer = models[selected_model_name]["tokenizer"]
            model = models[selected_model_name]["model"]

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

            current_user.tokens -= 1
            db.session.commit()

    return render_template('index.html', wynik=wynik, kolor=kolor,
                           perplexity=perplexity, burstiness=burstiness,
                           text=text, model_name=selected_model_name,
                           tokens=current_user.tokens)

# Uruchomienie aplikacji
if __name__ == '__main__':
    create_admin()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
