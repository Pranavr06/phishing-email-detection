# AI-Driven Phishing Email Detection Using NLP

## Overview
This repository contains the design, implementation, and evaluation of a machine-learning system for detecting phishing emails using Natural Language Processing (NLP) and structural feature engineering. 

During evaluation, additional dataset auditing and external validation revealed significant limitations in the generalizability of the trained models. While the models achieved perfect metrics on the provided synthetic training corpus, out-of-distribution manual testing demonstrated how structural dataset artifacts can inflate benchmark performance.

## Deployed Application (Educational Demonstration)
A Streamlit web application is deployed to demonstrate the model's behavior interactively:
**[Launch Phishing Detector](https://phishing-email-detection-nlp.streamlit.app)**

> [!WARNING]
> **Educational / Research Demonstration Only**
> This application does not guarantee email safety and should not replace security tools or human judgment. ML-based phishing detection can produce false positives and false negatives. Do not submit confidential or sensitive email contents. No data is stored or logged.

## Problem Statement
The objective was to design a robust machine-learning system to classify emails as Phishing (1) or Legitimate (0) by leveraging textual analysis (TF-IDF) alongside structural indicators (URLs, HTML tags, punctuation, and urgency keywords), while comparing multiple classification models.

## Key Features & ML Workflow
1. **Data Preprocessing**: Handling missing values, tokenization, and vectorization.
2. **Feature Engineering**: Custom scikit-learn transformers extracting structural markers (`num_real_urls`, `num_masked_urls`, `num_html_tags`, etc.).
3. **Model Evaluation**: Comparing Logistic Regression, Naive Bayes, Random Forest, and a Multi-Layer Perceptron (MLP).
4. **Dataset Auditing**: Investigating duplicate templates and dataset artifacts.
5. **Deployment**: Serialized inference pipeline integrated into an interactive UI.

## Dataset & Auditing Findings
The project utilized the `Zero-Day_Phishing_Emails_Corpus.csv` dataset.
- **Duplicates**: Only 0.53% of complete Subject+Body records were exact duplicates, but approximately 67.76% of records contained repeated subject lines, indicating substantial reuse of subject templates.
- **URL Artifacts**: No standard real URLs matching the implemented extraction pattern were detected in the legitimate class of the analyzed dataset. Masked `[URL]` tokens were distributed evenly.

## Model Evaluation
All four evaluated classifiers (Logistic Regression, Random Forest, Naive Bayes, MLPClassifier) achieved perfect performance (1.000 F1/Accuracy) on the provided dataset, even when using a Group-Aware split to prevent subject template leakage.

Logistic Regression was the selected deployment model due to its equivalent measured predictive performance, lower complexity, lightweight model artifact (0.26 MB), fast inference, and deployment simplicity.

## Manual External Validation
To assess true generalization, a small, manual external validation set of 8 representative examples was evaluated. 
- **Result**: The model correctly classified 3 of 8 cases (37.5%). 
- **Note**: This small manual validation set is diagnostic and illustrative, not a statistically representative estimate of real-world accuracy. It revealed that legitimate emails containing real URLs or urgency language were consistently misclassified as phishing due to the structural artifacts learned from the training data.

## Repository Structure
```
phishing-email-detection/
├── app.py                      # Streamlit application
├── data/
│   └── raw/                    # Original dataset
├── models/                     # Serialized .joblib and metadata
├── notebooks/
│   └── phishing_email_detection.ipynb  # Reproducible research notebook
├── reports/
│   ├── comparative_report.md   # Final academic analysis report
│   └── presentation_slides.md  # Presentation slide outline
├── requirements.txt            # Python dependencies
└── src/
    ├── data/                   # Data loading and integrity checks
    ├── features/               # Custom sklearn transformers
    └── models/                 # Training, evaluation, and serialization scripts
```

## Setup and Installation

### Running Locally
Due to Windows Application Control policies on some environments, running the environment via Google Colab is recommended for training. To run the UI locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/phishing-email-detection.git
   cd phishing-email-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

### Training the Model
To re-run the experiments and serialize the final model, execute the module:
```bash
python -m src.models.serialize
python -m src.models.manual_validation
```

## Limitations & Ethical Considerations
- **Dataset Bias**: The model heavily associates standard URLs and HTML with malicious intent due to their absence in the legitimate training class.
- **Real-World Viability**: High in-distribution benchmark performance alone is insufficient evidence of real-world phishing-detection capability.
- **Privacy**: The local application does not log user inputs, but deployed iterations must secure transmission of potentially sensitive email contents.

## Future Improvements
Future work should prioritize:
- A more representative dataset containing legitimate emails with URLs, HTML, forms, deadlines, and urgency language.
- Phishing examples without obvious URLs or urgency cues.
- Deduplication and template-aware dataset construction.
- Larger independent external validation sets.
