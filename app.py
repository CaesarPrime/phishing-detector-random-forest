"""
app.py - Phishing Detector System Using Random Forest (web application)
Run:  py -3.10 app.py   then open http://127.0.0.1:5000
Requires model.joblib produced by train.py (or the Kaggle notebook).
"""
import io, base64, os, sqlite3
from datetime import datetime
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

from features import FEATURE_NAMES, feature_vector, rule_analyse

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "phishing_logs.db")
MODEL_PATH = os.path.join(BASE, "model.joblib")
PHISHING, LEGIT = "Phishing URL Detected", "Legitimate URL"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)          # trusted file created by train.py only


def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS logs (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     url TEXT NOT NULL,
                     result TEXT NOT NULL,
                     date_checked TEXT NOT NULL)""")
    return con


def detect(url):
    score, rule_phish, reasons = rule_analyse(url)              # raises ValueError if invalid
    X = pd.DataFrame([feature_vector(url)], columns=FEATURE_NAMES)
    model_phish = bool(model.predict(X)[0])
    confidence = float(model.predict_proba(X)[0][1]) * 100
    final = PHISHING if (rule_phish or model_phish) else LEGIT   # decision fusion
    return {
        "url": url.strip(), "result": final,
        "rule_result": PHISHING if rule_phish else LEGIT,
        "model_result": PHISHING if model_phish else LEGIT,
        "score": score, "reasons": reasons, "confidence": round(confidence, 1),
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/check", methods=["POST"])
def check():
    url = request.form.get("url", "")
    try:
        r = detect(url)
    except ValueError as e:
        return render_template("index.html", error=str(e), url=url)
    with db() as con:   # parameterised query prevents SQL injection
        con.execute("INSERT INTO logs (url, result, date_checked) VALUES (?, ?, ?)",
                    (r["url"], r["result"], datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return render_template("result.html", r=r)


@app.route("/history")
def history():
    with db() as con:
        rows = con.execute("SELECT id, url, result, date_checked FROM logs ORDER BY id DESC").fetchall()
    return render_template("history.html", rows=rows, phishing=PHISHING)


@app.route("/dashboard")
def dashboard():
    with db() as con:
        total = con.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        phish = con.execute("SELECT COUNT(*) FROM logs WHERE result = ?", (PHISHING,)).fetchone()[0]
    legit = total - phish
    chart = None
    if total:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie([phish, legit], labels=["Phishing", "Legitimate"], autopct="%1.1f%%", startangle=90)
        ax.set_title("Phishing Detection Statistics")
        buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
        chart = base64.b64encode(buf.getvalue()).decode()
    return render_template("dashboard.html", total=total, phish=phish, legit=legit, chart=chart)


if __name__ == "__main__":
    db().close()
    app.run(host="127.0.0.1", port=5000, debug=False)
