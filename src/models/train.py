import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from src.features.build_features import (
    URLFeatureExtractor,
    HTMLFeatureExtractor,
    PunctuationFeatureExtractor,
    UrgencyFeatureExtractor
)
from src.data.preprocess import TextPreprocessor

def build_training_pipeline(classifier_type="logistic_regression", stop_words=None, remove_masked_urls=False):
    """
    Constructs the end-to-end inference and training pipeline.
    """
    
    # We can conditionally remove the URLFeatureExtractor if ablating masked urls
    # But URLFeatureExtractor returns BOTH num_real_urls and num_masked_urls.
    # To ablate, we could either drop the column in a custom transformer or modify URLFeatureExtractor.
    # For simplicity, we pass remove_masked_urls to a column dropper or just leave it for now
    # Wait, the simplest way is to pass a flag to URLFeatureExtractor or drop it later.
    # Let's add a custom column dropper if remove_masked_urls is True.
    # But actually, URLFeatureExtractor doesn't take that flag right now. Let's just drop the column.
    
    structural_features = Pipeline([
        ('extractors', FeatureUnion([
            ('url_features', URLFeatureExtractor(text_column='Body', remove_masked_urls=remove_masked_urls)),
            ('html_features', HTMLFeatureExtractor(text_column='Body')),
            ('punctuation', PunctuationFeatureExtractor(text_column='Body')),
            ('urgency', UrgencyFeatureExtractor(subject_col='Subject', body_col='Body'))
        ])),
        # MaxAbsScaler preserves sparsity and avoids negative values.
        # Since all our structural features are counts (>=0), output remains >= 0 (compatible with MultinomialNB).
        ('scaler', MaxAbsScaler())
    ])
    
    # Baseline TF-IDF configuration:
    # ngram_range=(1,2) to capture phrases like "action required".
    # stop_words is configurable (default None).
    text_features = Pipeline([
        ('cleaner', TextPreprocessor(subject_col='Subject', body_col='Body')),
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words=stop_words, ngram_range=(1, 2), sublinear_tf=True, min_df=5))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('structural', structural_features, ['Subject', 'Body']),
            ('text', text_features, ['Subject', 'Body'])
        ]
    )
    
    if classifier_type == "random_forest":
        classifier = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100, max_depth=20) # Conservative depth to save memory on 5000+ sparse features
    elif classifier_type == "naive_bayes":
        classifier = MultinomialNB()
    elif classifier_type == "mlp":
        # simple baseline MLP to avoid OOM
        classifier = MLPClassifier(hidden_layer_sizes=(64,), max_iter=50, random_state=42, early_stopping=True)
    else:
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline
