# Phishing Detector System Using Random Forest

A Flask web application that classifies a URL as **phishing** or **legitimate**. It uses a Random Forest model together with a rule-based analyser, stores every check in SQLite, and shows the detection history and a dashboard.

## Files
| File | Purpose |
|---|---|
| `features.py` | URL feature extraction (12 features) and the rule-based analyser. The same code is used for training and for live detection. |
| `train.py` | Trains and evaluates the Random Forest model. Writes `model.joblib`, `results.json` and two figures. |
| `app.py` | The Flask web application (home, result, history and dashboard pages). |
| `templates/`, `static/` | The HTML pages and the stylesheet. |
| `phishing_random_forest_kaggle.ipynb` | A Kaggle notebook that runs the training and prints every value needed for Chapter Four. |

## Step 1: get the real results on Kaggle
1. Sign in at kaggle.com, then go to **Create → New Notebook → File → Import Notebook** and upload `phishing_random_forest_kaggle.ipynb`.
2. In the right-hand panel, click **Add Input** and search for **"Web page Phishing Detection Dataset"**. This is Hannousse & Yahiouche (2021): 11,430 URLs in the file `dataset_phishing.csv`. Click **Add**.
3. Click **Run All**. It takes about a minute.
4. Copy the printed `results.json` from the last cell. From the **Output** tab, download `model.joblib`, `confusion_matrix.png` and `feature_importance.png`.

## Step 2: run the web application on Windows
```
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Put `model.joblib` (from Kaggle) in this folder, then run:
```
py -3.10 app.py
```
Open http://127.0.0.1:5000 in your browser.

If you prefer, you can train locally instead of on Kaggle: `python train.py dataset_phishing.csv`

## Notes
- The website being checked is never visited. Only the text of the URL is analysed.
- Decision fusion: a URL is reported as phishing if **either** the rules **or** the model flags it. `results.json` reports the model alone, the rules alone and the combined (hybrid) result, so the effect of this choice can be discussed in Section 4.7.
- Use a random state of 42, 100 trees and an 80:20 stratified split, so the results can be reproduced.
