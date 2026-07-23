# Comparative Report: Generalization Analysis of Perfect In-Dataset Performance in Phishing Detection

## 1. Abstract
This report details the design, implementation, and evaluation of a machine-learning system for detecting phishing emails. While four evaluated classifiers achieved perfect held-out performance (1.000 F1) on the provided corpus, subsequent dataset auditing and external manual validation revealed significant limitations in the generalizability of the trained models. Only 3 of 8 representative unseen examples were classified correctly in an illustrative out-of-distribution evaluation. Dataset analysis identified synthetic structural artifacts—including unrealistic URL distributions—as likely shortcut features. These findings demonstrate that high in-distribution benchmark performance alone is insufficient evidence of real-world phishing-detection capability.

## 2. Introduction
Phishing remains a prevalent vector for cyberattacks. Machine learning models, particularly those leveraging Natural Language Processing (NLP), are frequently deployed to detect malicious intent in email text. This project explores the effectiveness of combining TF-IDF lexical features with engineered structural markers.

## 3. Problem Statement
The objective was to design a robust machine-learning system to classify emails as Phishing (1) or Legitimate (0) by leveraging textual analysis and structural features, while comparing multiple classification models and rigorously evaluating their true generalization capability.

## 4. Objectives
- Design and implement a machine-learning system that detects phishing emails using NLP and structural features.
- Compare multiple classification models (Logistic Regression, Random Forest, Naive Bayes, MLP).
- Investigate perfect benchmark performance through dataset auditing and external validation.

## 5. Dataset Description and Provenance
The experiments utilized the `Zero-Day_Phishing_Emails_Corpus.csv` dataset, comprising 60,000 labeled email records. The dataset contains `Subject` and `Body` fields. Provenance analysis indicated the dataset was synthetically generated or heavily templated.

## 6. Dataset Integrity Audit
An integrity audit was performed to assess template reuse and leakage risk:
- **Total records:** 60,000
- **Exact duplicates:** Only 0.53% of complete Subject+Body records were exact duplicates.
- **Template reuse:** Approximately 67.76% of records contained repeated subject lines, indicating substantial reuse of subject templates.
To prevent data leakage, a Group-Aware splitting strategy (grouping by normalized Subject) was necessary.

## 7. Preprocessing
Text preprocessing was implemented conservatively to preserve malicious signals often removed by standard NLP pipelines:
- Missing values (np.nan) were replaced with empty strings.
- Stop words were not removed (e.g., `stop_words=None`) for the baseline models, as common words often form the grammatical structure of social engineering attempts.
- N-grams were extracted using a range of (1, 2).

## 8. Feature Engineering
Custom scikit-learn transformers were developed to extract structural metadata from the raw email body before vectorization:
- **URL Features:** Extracted standard URLs (http/https/www) and masked `[URL]` tokens.
- **HTML Features:** Counted total tags and specific anchor tags.
- **Punctuation:** Calculated uppercase ratios and counted specific symbols (!, ?).
- **Urgency:** Counted exact matches of urgency keywords (e.g., "urgent", "immediate").

## 9. Model Development
A unified scikit-learn `Pipeline` and `FeatureUnion` architecture was utilized to seamlessly combine the sparse TF-IDF matrix with the dense engineered features. A `MaxAbsScaler` was applied to normalize the combined feature space for distance-based and gradient-based classifiers.

## 10. Experimental Methodology
Four required baseline model families were evaluated:
- Logistic Regression
- Random Forest
- Naive Bayes (MultinomialNB)
- Simple Neural Network (MLPClassifier)

## 11. Evaluation Metrics
Models were evaluated using Accuracy, Precision, Recall, and F1-Score on a held-out test set. 

## 12. Baseline Model Comparison
**IN-DISTRIBUTION RESULTS**
All four evaluated classifiers achieved equivalent metrics of 1.000 across Accuracy, Precision, Recall, and F1-Score on the provided dataset. Confusion matrices recorded zero false positives and zero false negatives on the test set.

Logistic Regression was the selected deployment model based on its equivalent measured predictive performance, lower complexity, lightweight model artifact (0.26 MB), fast inference, and deployment simplicity.

## 13. Random vs Group-Aware Evaluation
To test if perfect performance was an artifact of template leakage across the train/test boundary, the models were re-evaluated using a `GroupShuffleSplit` on the normalized Subject lines. The models maintained 1.000 F1-Scores, proving the models were not merely memorizing exact templates, but rather learning underlying structural patterns that perfectly separated the classes.

## 14. Artifact / Ablation Analysis
Perfect confusion matrices across multiple model families prompted further investigation, because such uniformly perfect performance is unusual for realistic phishing-email classification. 
Analysis of structural features revealed a critical distribution artifact:
- **No standard real URLs matching the implemented extraction pattern were detected in the legitimate class of the analyzed dataset.**
An ablation experiment removing the masked `[URL]` feature resulted in identical perfect performance, confirming the models were heavily leveraging the presence of standard URLs and HTML as shortcut features.

## 15. Manual External Validation
**EXTERNAL/MANUAL VALIDATION**
To assess true generalizability, a manual external validation set of 8 representative examples was evaluated through the serialized inference pipeline.
- **Outcome:** The model correctly classified 3 of 8 cases (37.5%).
- **Note:** This small manual validation set is diagnostic and illustrative, not a statistically representative estimate of real-world accuracy.
Legitimate emails containing URLs or HTML were confidently misclassified as phishing (96%-100% probability). Phishing emails lacking URLs were occasionally misclassified as legitimate.

## 16. Genuine Institutional Email Case Study
A genuine legitimate institutional email was tested as a qualitative case study:
- **Subject:** "Update on Salesforce Administrator Course Progress - 2028 Batch"
- **Content:** Contained a Google Forms URL, deadline language, and mandatory-action wording.
The model incorrectly classified this email as phishing. This demonstrates a practical false-positive scenario: legitimate institutional emails naturally contain URLs, deadlines, forms, and urgency-like wording. Because the synthetic legitimate training data lacked these features, the model received a distorted view of reality.

## 17. Discussion
The discrepancy between the in-distribution benchmarks and the out-of-distribution manual evaluation highlights the risks of synthetic datasets. The models achieved high scores not by understanding phishing intent, but by exploiting superficial structural absences in the legitimate class.

## 18. Limitations
The primary limitation of this system is the dataset bias. The deployed application is an educational demonstration and does not guarantee email safety. It should not replace security tools or human judgment.

## 19. Future Work
Future work must address the data quality bottleneck. Priority improvements include:
- A more representative dataset containing legitimate emails with URLs, HTML, forms, deadlines, and urgency language.
- Phishing examples without obvious URLs or urgency cues.
- Realistic sender/domain metadata where legally and ethically available.
- Deduplication and template-aware dataset construction.
- Larger independent external validation sets.
Larger language models (e.g., Transformers) should only be considered after dataset quality is improved.

## 20. Conclusion
All four evaluated classifiers achieved perfect performance on the provided dataset under the tested evaluation configurations. However, dataset auditing identified substantial template reuse and structural artifacts, while a small illustrative external manual validation set revealed significant generalization failures, particularly for legitimate emails containing URLs and urgency-like language. These findings demonstrate that high benchmark performance alone is insufficient evidence of robust real-world phishing detection. Future improvements should prioritize representative data collection, cross-dataset evaluation, and stronger external validation before considering production deployment.
