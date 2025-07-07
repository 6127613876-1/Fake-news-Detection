# app.py

from flask import Flask, request, render_template_string
import pandas as pd
import os
import nltk
import time
import threading
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Configuration ===
app = Flask(__name__)
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)
for pkg in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_path)

# === Globals ===
model = None
TOKENIZER = None
news_data = {}
max_length = 200
similarity_threshold = 0.7

# === Google Gemini ===
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# === HTML Template ===
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>News Real or Fake Checker</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css">
</head>
<body>
  <div class="container mt-5">
    <h1 class="text-center text-primary">News Real or Fake Checker</h1>
    <form method="POST" action="/check">
      <div class="mb-3">
        <label for="news" class="form-label">Enter News Headline</label>
        <input type="text" id="news" name="news" class="form-control" required>
      </div>
      <button type="submit" class="btn btn-success">Check</button>
    </form>
    {% if result %}
    <div class="alert alert-info mt-4" role="alert">
      <strong>Result:</strong> {{ result }}
    </div>
    {% endif %}
  </div>
</body>
</html>
"""

# === Web Crawler Sources ===
WEB_CONFIG = {
    "BBC": {
        "url": "https://www.bbc.com/news",
        "content_selector": 'h2[data-testid="card-headline"]',
        "publisher": "BBC"
    },
    "CNN": {
        "url": "https://edition.cnn.com/world",
        "content_selector": 'span.container__headline-text',
        "publisher": "CNN"
    },
    "The Hindu": {
        "url": "https://www.thehindu.com/",
        "content_selector": 'strong',
        "publisher": "The Hindu"
    }
}

# === Load Model & Tokenizer ===
def load_dependencies():
    global model, TOKENIZER
    model = load_model("lstm.h5")
    df = pd.read_csv("news_dataset.csv")
    TOKENIZER = Tokenizer(num_words=5000)
    TOKENIZER.fit_on_texts(df["Text"].astype(str).values)

# === Preprocessing ===
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    seq = TOKENIZER.texts_to_sequences([" ".join(tokens)])
    return pad_sequences(seq, maxlen=max_length)

# === Model Prediction ===
def predict_news(text):
    seq = preprocess_text(text)
    pred = model.predict(seq)
    print("Prediction scores:", pred)  # Debug output
    return "Real" if pred.argmax(axis=1)[0] == 1 else "Fake"

# === Gemini Comparison ===
def verify_with_gemini(original, matched):
    prompt = f"""
Compare the following two news headlines and check if they are semantically and factually the same.
Pay close attention to numbers, names, and specific facts. Reply only with 'Yes' or 'No'.

1. {original}
2. {matched}
"""
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini Error:", e)
        return "error"

# === Find Matching News ===
def find_similar_news(text):
    all_news = [item for df in news_data.values() for item in df["Content"].values]
    if not all_news:
        return None
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(all_news + [text])
    similarities = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    max_sim = similarities.max()
    if max_sim >= similarity_threshold:
        return all_news[similarities.argmax()]
    return None

# === News Crawling ===
def crawl_news(source):
    global news_data
    config = WEB_CONFIG[source]
    try:
        html = requests.get(config["url"], timeout=10).content
        soup = BeautifulSoup(html, "html.parser")
        elements = soup.select(config["content_selector"])
        data = []
        for i, el in enumerate(elements[:20]):
            txt = el.get_text(strip=True)
            if txt:
                data.append({
                    "S.No": i + 1,
                    "Content": txt,
                    "Publisher": config["publisher"],
                    "Date": time.strftime("%Y-%m-%d")
                })
        news_data[source] = pd.DataFrame(data)
    except Exception as e:
        print(f"Error crawling {source}: {e}")

# === Background News Fetching ===
def schedule_news_crawling():
    while True:
        for source in WEB_CONFIG:
            print(f"Fetching news from: {source}")
            crawl_news(source)
        time.sleep(24 * 3600)

# === Flask Routes ===
@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE, result=None)

@app.route("/check", methods=["POST"])
def check_news():
    user_text = request.form["news"]
    matched = find_similar_news(user_text)

    if matched:
        gemini_verdict = verify_with_gemini(user_text, matched)
        if gemini_verdict.lower().strip() == "no":
            result = f"Fake (Gemini disagreed: \"{matched[:100]}...\")"
        elif gemini_verdict.lower().strip() == "yes":
            result = f"Real (Matched with: \"{matched[:100]}...\")"
        else:
            result = f"Uncertain (Gemini error or unclear response: \"{gemini_verdict}\")"
    else:
        result = predict_news(user_text) + " (No similar headline found)"

    return render_template_string(HTML_TEMPLATE, result=result)

@app.route("/health")
def health_check():
    return "OK", 200

# === Start App ===
if __name__ == "__main__":
    load_dependencies()
    threading.Thread(target=schedule_news_crawling, daemon=True).start()
    app.run(debug=True, port=5000)
