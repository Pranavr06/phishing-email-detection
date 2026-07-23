import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# AI-Driven Phishing Email Detection\n\nThis notebook demonstrates the complete workflow for detecting phishing emails, from dataset integrity auditing to feature engineering, baseline model comparison, and out-of-distribution manual validation."),
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport joblib\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Configure plotting\nplt.style.use('seaborn-v0_8-darkgrid')"),
    nbf.v4.new_markdown_cell("## 1. Dataset Loading and Overview"),
    nbf.v4.new_code_cell("df = pd.read_csv('../data/raw/Zero-Day_Phishing_Emails_Corpus.csv')\ndf.head()"),
    nbf.v4.new_code_cell("print(f'Total records: {len(df)}')\nprint(df['Label'].value_counts(normalize=True))"),
    nbf.v4.new_markdown_cell("## 2. Dataset Integrity Checks\nOur provenance analysis indicated potential synthetic generation or template reuse. Let's quantify this."),
    nbf.v4.new_code_cell("from src.data.dataset_audit import audit_duplicates\naudit_duplicates(df)"),
    nbf.v4.new_markdown_cell("## 3. Feature Engineering\nWe extract structural markers (URLs, HTML, Punctuation, Urgency) prior to vectorizing the text."),
    nbf.v4.new_code_cell("from src.features.build_features import (\n    URLFeatureExtractor, HTMLFeatureExtractor,\n    PunctuationFeatureExtractor, UrgencyFeatureExtractor\n)\n\n# Example of URL extraction on a small subset\nsubset = df.sample(5, random_state=42)\nurl_ext = URLFeatureExtractor()\nurl_features = url_ext.fit_transform(subset)\nurl_features"),
    nbf.v4.new_markdown_cell("## 4. Train/Test Methodology and Model Training\nBecause ~67.7% of the dataset shares subject templates, we use a `GroupShuffleSplit` (group-aware split) on normalized subjects to prevent data leakage across the train/test boundary."),
    nbf.v4.new_code_cell("from src.models.train import build_training_pipeline\nfrom sklearn.model_selection import GroupShuffleSplit\nimport re\n\n# Creating normalized subjects for grouping\ndef normalize_subject(subj):\n    s = str(subj).lower().strip()\n    return re.sub(r'[^a-z0-9]', '', s)\n\ndf['Norm_Subject'] = df['Subject'].apply(normalize_subject)\n\ngss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)\ntrain_idx, test_idx = next(gss.split(df, groups=df['Norm_Subject']))\n\nX_train, y_train = df.iloc[train_idx][['Subject', 'Body']], df.iloc[train_idx]['Label']\nX_test, y_test = df.iloc[test_idx][['Subject', 'Body']], df.iloc[test_idx]['Label']\n\nprint(f'Training set size: {len(X_train)}\\nTest set size: {len(X_test)}')"),
    nbf.v4.new_markdown_cell("Let's train the selected deployment model: Logistic Regression."),
    nbf.v4.new_code_cell("pipe = build_training_pipeline('logistic_regression', stop_words=None)\npipe.fit(X_train, y_train)"),
    nbf.v4.new_markdown_cell("## 5. Evaluation Metrics and Confusion Matrix\nLet's evaluate the model on the held-out test set."),
    nbf.v4.new_code_cell("from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay\n\npreds = pipe.predict(X_test)\nprint(classification_report(y_test, preds))\n\ncm = confusion_matrix(y_test, preds)\ndisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Phishing'])\ndisp.plot(cmap='Blues')\nplt.title('Confusion Matrix on Test Set')\nplt.show()"),
    nbf.v4.new_markdown_cell("## 6. Artifact Analysis\nPerfect confusion matrices across multiple model families prompted further investigation, because such uniformly perfect performance is unusual for realistic phishing-email classification. Let's look at the distribution of real URLs."),
    nbf.v4.new_code_cell("all_url_features = url_ext.fit_transform(df)\nurl_analysis = pd.concat([df['Label'], all_url_features], axis=1)\nurl_analysis.groupby('Label')[['num_real_urls', 'num_masked_urls']].mean()"),
    nbf.v4.new_markdown_cell("> **Finding:** No standard real URLs matching the implemented extraction pattern were detected in the legitimate class of the analyzed dataset. The model learned to use the presence of a URL as a near-perfect shortcut feature for Phishing."),
    nbf.v4.new_markdown_cell("## 7. Manual External Validation\nTo assess true generalizability, a manual external validation set of representative examples was evaluated through the serialized inference pipeline."),
    nbf.v4.new_code_cell("from src.models.manual_validation import run_manual_validation\n\n# Run the manual validation script which tests 8 carefully crafted examples\nrun_manual_validation()"),
    nbf.v4.new_markdown_cell("### Genuine Institutional Email Case Study\nWe tested a genuine legitimate institutional email:\n\n*\"Update on Salesforce Administrator Course Progress - 2028 Batch... Please note that the deadline to complete the Salesforce Administrator Course is 15th July 2026. It's compulsory...\"*\n\nThe model incorrectly classified this email as phishing. This demonstrates a practical false-positive scenario: legitimate institutional emails naturally contain URLs, deadlines, forms, and urgency-like wording. Because the synthetic legitimate training data lacked these features, the model received a distorted view of reality."),
    nbf.v4.new_markdown_cell("## 8. Final Conclusions\nAll four evaluated classifiers achieved perfect performance on the provided dataset under the tested evaluation configurations. However, dataset auditing identified substantial template reuse and structural artifacts, while a small illustrative external manual validation set revealed significant generalization failures, particularly for legitimate emails containing URLs and urgency-like language. \n\nThese findings demonstrate that high benchmark performance alone is insufficient evidence of robust real-world phishing detection. Future improvements should prioritize representative data collection, cross-dataset evaluation, and stronger external validation before considering production deployment.")
]

nb['cells'] = cells

os.makedirs("notebooks", exist_ok=True)
with open('notebooks/phishing_email_detection.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated.")
