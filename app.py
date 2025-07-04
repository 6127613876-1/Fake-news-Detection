from flask import Flask, request, render_template_string
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import threading
import time
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

# NLTK setup (avoid redownloads)
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)
for pkg in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_path)

app = Flask(__name__)

# Globals
news_data = {}
model = None
TOKENIZER = None
max_length = 5000
similarity_threshold = 0.7

# Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# HTML Template
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

def load_dependencies():
    global model, TOKENIZER
    model = load_model("lstm.h5")
    data = pd.read_csv("news_dataset.csv")
    if "Text" not in data.columns:
        raise ValueError("CSV must have 'Text' column.")
    TOKENIZER = Tokenizer(num_words=5000)
    TOKENIZER.fit_on_texts(data["Text"].values)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    sequence = TOKENIZER.texts_to_sequences([" ".join(filtered_tokens)])
    return pad_sequences(sequence, maxlen=max_length)

def predict_news(text):
    processed = preprocess_text(text)
    prediction = model.predict(processed)
    return "Real" if prediction.argmax(axis=1)[0] == 1 else "Fake"

def find_similar_news(input_text):
    all_news = [item for df in news_data.values() for item in df["Content"].values]
    if not all_news:
        return None
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(all_news + [input_text])
    similarities = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    max_sim = similarities.max()
    if max_sim >= similarity_threshold:
        return all_news[similarities.argmax()]
    return None

def verify_with_gemini(input_text, matched_text):
    prompt = f"""
    Compare the following two news headlines and tell me if they mean the same thing. Reply with only 'Yes' or 'No'.

    1. {input_text}
    2. {matched_text}
    """
    response = genai.GenerativeModel('gemini-pro').generate_content(prompt)
    return response.text.strip()

def crawl_news(website):
    global news_data
    config = WEB_CONFIG[website]
    try:
        response = requests.get(config["url"], timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        elements = soup.select(config["content_selector"])
        data = []
        for idx, el in enumerate(elements[:20]):
            content = el.get_text(strip=True)
            if content:
                data.append({
                    "S.No": idx + 1,
                    "Content": content,
                    "Publisher": config["publisher"],
                    "Date": time.strftime("%Y-%m-%d")
                })
        news_data[website] = pd.DataFrame(data)
    except Exception as e:
        print(f"Error crawling {website}: {e}")

def schedule_updates():
    while True:
        for site in WEB_CONFIG:
            print(f"Crawling {site}...")
            crawl_news(site)
        time.sleep(24 * 3600)

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, result=None)

@app.route("/check", methods=["POST"])
def check_news():
    news = request.form["news"]
    similar = find_similar_news(news)
    if similar:
        gemini_result = verify_with_gemini(news, similar)
        if gemini_result.lower() == "no":
            result = f"Fake (Gemini disagreed with match: '{similar[:100]}...')"
        else:
            result = f"Real (Matched: '{similar[:100]}...')"
    else:
        prediction = predict_news(news)
        result = f"{prediction} (No match found)"
    return render_template_string(HTML_TEMPLATE, result=result)

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    load_dependencies()
    threading.Thread(target=schedule_updates, daemon=True).start()
    app.run(debug=True, port=5000)
