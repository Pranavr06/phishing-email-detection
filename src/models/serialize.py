import os
import json
import joblib
import datetime
import pandas as pd
import sklearn
import sys

from src.models.train import build_training_pipeline

def serialize_model():
    print("Loading data...")
    df = pd.read_csv("data/raw/Zero-Day_Phishing_Emails_Corpus.csv")
    X = df[['Subject', 'Body']]
    y = df['Label']

    print("Building and fitting Logistic Regression pipeline on full dataset...")
    # Using Logistic Regression as it's the fastest and achieved equal (1.0) performance
    pipe = build_training_pipeline("logistic_regression", stop_words=None, remove_masked_urls=False)
    pipe.fit(X, y)

    print("Saving pipeline...")
    os.makedirs("models", exist_ok=True)
    model_path = "models/final_pipeline.joblib"
    joblib.dump(pipe, model_path)
    
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model saved to {model_path}. Size: {size_mb:.2f} MB")

    print("Saving metadata...")
    metadata = {
        "model_type": "Logistic Regression Pipeline",
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_name": "Zero-Day_Phishing_Emails_Corpus.csv",
        "number_of_training_samples": len(df),
        "tfidf_configuration": {
            "max_features": 5000,
            "ngram_range": [1, 2],
            "stop_words": None,
            "sublinear_tf": True,
            "min_df": 5
        },
        "structural_features": [
            "num_real_urls", "num_masked_urls", "num_urls", "num_unique_real_urls",
            "num_html_tags", "has_html", "num_anchor_tags",
            "num_exclamation_marks", "num_question_marks", "num_currency_symbols",
            "uppercase_ratio", "urgency_keyword_count", "has_urgency"
        ],
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
        "evaluation_summary": "F1=1.000, Recall=1.000, 0 FN (Baseline on Synthetic Data)",
        "known_dataset_limitations": "The evaluated models achieved perfect classification on the provided dataset. Dataset auditing revealed strong synthetic and structural artifacts (e.g. lack of real URLs in legitimate class), so these results should not be interpreted as evidence of equivalent real-world performance."
    }
    
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Testing serialization round-trip...")
    loaded_pipe = joblib.load(model_path)
    
    # Test on a small subset
    sample = X.iloc[[0, 100, -1]]
    pred_orig = pipe.predict(sample)
    pred_loaded = loaded_pipe.predict(sample)
    
    assert (pred_orig == pred_loaded).all(), "Serialization round-trip failed!"
    print("Round-trip test passed successfully.")

if __name__ == "__main__":
    serialize_model()
