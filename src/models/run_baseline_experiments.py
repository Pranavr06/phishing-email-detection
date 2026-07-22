import os
import time
import pandas as pd
import numpy as np
import traceback
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.models.train import build_training_pipeline
from src.features.build_features import URLFeatureExtractor

def analyze_url_artifact(df):
    """Calculates [URL] artifact statistics by class and saves to CSV."""
    print("\n--- [URL] ARTIFACT ANALYSIS ---")
    
    url_ext = URLFeatureExtractor()
    url_features = url_ext.transform(df)
    
    analysis_df = pd.concat([url_features, df[['Label']]], axis=1)
    
    phishing = analysis_df[analysis_df['Label'] == 1]
    legit = analysis_df[analysis_df['Label'] == 0]
    
    stats = []
    
    pct_phishing_masked = (phishing['num_masked_urls'] > 0).mean() * 100
    pct_legit_masked = (legit['num_masked_urls'] > 0).mean() * 100
    
    avg_masked_phishing = phishing['num_masked_urls'].mean()
    avg_masked_legit = legit['num_masked_urls'].mean()
    
    avg_real_phishing = phishing['num_real_urls'].mean()
    avg_real_legit = legit['num_real_urls'].mean()
    
    stats.append({"Class": "Phishing", "Pct_With_Masked_URL": pct_phishing_masked, "Avg_Masked_URLs": avg_masked_phishing, "Avg_Real_URLs": avg_real_phishing})
    stats.append({"Class": "Legitimate", "Pct_With_Masked_URL": pct_legit_masked, "Avg_Masked_URLs": avg_masked_legit, "Avg_Real_URLs": avg_real_legit})
    
    stats_df = pd.DataFrame(stats)
    os.makedirs('results', exist_ok=True)
    stats_df.to_csv('results/artifact_analysis.csv', index=False)
    print("Saved to results/artifact_analysis.csv")
    print(stats_df)
    
    return stats_df

def run_experiments():
    file_path = "data/raw/Zero-Day_Phishing_Emails_Corpus.csv"
    if not os.path.exists(file_path):
        print(f"Dataset not found at {file_path}. Please upload it before running.")
        return
        
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    analyze_url_artifact(df)
    
    # 1. Create normalized subject for grouping
    df['normalized_subject'] = df['Subject'].astype(str).str.lower().str.strip()
    
    X = df[['Subject', 'Body']]
    y = df['Label']
    groups = df['normalized_subject']
    
    # 2. Split Strategies
    print("\n--- SPLITTING ---")
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    random_train_idx, random_test_idx = next(sss.split(X, y))
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    group_train_idx, group_test_idx = next(gss.split(X, y, groups=groups))
    
    models = ["logistic_regression", "naive_bayes", "random_forest", "mlp"]
    all_metrics = []
    
    def evaluate_model(model_name, split_type, train_idx, test_idx, remove_masked=False):
        print(f"Training {model_name} (Split: {split_type}, Ablation: {remove_masked})...")
        try:
            start_train = time.time()
            # Test stop_words=None as main baseline (per instructions)
            pipe = build_training_pipeline(model_name, stop_words=None, remove_masked_urls=remove_masked)
            
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            
            pipe.fit(X_train, y_train)
            train_time = time.time() - start_train
            
            start_infer = time.time()
            y_pred = pipe.predict(X_test)
            infer_time = time.time() - start_infer
            
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            metrics = {
                "Model": model_name,
                "Split_Type": split_type,
                "Ablation_Remove_Masked": remove_masked,
                "Status": "SUCCESS",
                "Error": None,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1_Score": f1_score(y_test, y_pred),
                "False_Positives": int(fp),
                "False_Negatives": int(fn),
                "Training_Time_s": round(train_time, 2),
                "Inference_Time_s": round(infer_time, 2)
            }
            return metrics
        except Exception as e:
            print(f"FAILED: {model_name} on {split_type}. Error: {e}")
            return {
                "Model": model_name,
                "Split_Type": split_type,
                "Ablation_Remove_Masked": remove_masked,
                "Status": "FAILED",
                "Error": str(e),
                "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan, "F1_Score": np.nan,
                "False_Positives": np.nan, "False_Negatives": np.nan,
                "Training_Time_s": np.nan, "Inference_Time_s": np.nan
            }

    # Run Standard Models
    for model_name in models:
        # Experiment A: Random Split
        res_random = evaluate_model(model_name, "Random", random_train_idx, random_test_idx)
        all_metrics.append(res_random)
        
        # Experiment B: Group-Aware Split
        res_group = evaluate_model(model_name, "Group-Aware", group_train_idx, group_test_idx)
        all_metrics.append(res_group)
        
    # Artifact Ablation Experiment on Logistic Regression (Fastest, most interpretable baseline)
    print("\n--- ARTIFACT ABLATION EXPERIMENT ---")
    res_ablation = evaluate_model("logistic_regression", "Group-Aware", group_train_idx, group_test_idx, remove_masked=True)
    all_metrics.append(res_ablation)

    metrics_df = pd.DataFrame(all_metrics)
    os.makedirs('results', exist_ok=True)
    metrics_df.to_csv('results/baseline_model_comparison.csv', index=False)
    
    print("\nResults saved to results/baseline_model_comparison.csv")
    print(metrics_df[['Model', 'Split_Type', 'Ablation_Remove_Masked', 'Status', 'F1_Score', 'False_Negatives']])

if __name__ == "__main__":
    run_experiments()
