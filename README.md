# CEARS-Contextual-Emotion-Aware-Explainable-Recommendation-System
A hybrid AI-powered product recommendation system built with Python and Streamlit.

## What it does
CEARS recommends Amazon products based on three signals:
- **Content Similarity** — TF-IDF + cosine similarity
- **Budget Context** — maps user budget level to recommendation score
- **Mood & Emotion** — TextBlob sentiment analysis on user input

## Project Structure
| File | Purpose |
|------|---------|
| app.py | Main application — UI and recommendation pipeline |
| hybrid.py | Core ML engine — TF-IDF similarity scoring |
| context.py | Budget-context adjustment layer |
| emotion.py | Mood/emotion scoring using TextBlob |
| explain.py | Explainability — why each product was recommended |
| amazon.csv | Dataset — 1400+ Amazon products |

## Tech Stack
- Python 3.9+
- Streamlit — web UI
- Pandas — data processing
- scikit-learn — TF-IDF and cosine similarity
- TextBlob — sentiment analysis
- Plotly — charts and visualisations

## How to Run
pip install streamlit pandas scikit-learn textblob plotly
streamlit run app.py

## Dataset
Amazon product dataset with 1400+ real products across multiple
categories including electronics, accessories, and more.
