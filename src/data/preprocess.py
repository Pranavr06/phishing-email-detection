import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Conservatively cleans text for TF-IDF.
    Combines Subject and Body with explicit separators.
    Optionally strips HTML tags while preserving visible text.
    Lowercases text but DOES NOT blindly remove punctuation or stopwords.
    """
    def __init__(self, subject_col='Subject', body_col='Body', strip_html=True, lowercase=True):
        self.subject_col = subject_col
        self.body_col = body_col
        self.strip_html = strip_html
        self.lowercase = lowercase
        
        # Regex to match HTML tags
        self.html_pattern = re.compile(r'<[^>]+>')
        # Regex to normalize whitespace
        self.whitespace_pattern = re.compile(r'\s+')

    def fit(self, X, y=None):
        return self

    def _clean_text(self, text):
        if pd.isnull(text):
            return ""
        text = str(text)
        
        if self.strip_html:
            text = self.html_pattern.sub(' ', text)
            
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        if self.lowercase:
            text = text.lower()
            
        return text

    def transform(self, X):
        # We return a 1D array/series of strings for the TfidfVectorizer
        processed_texts = []
        
        for _, row in X.iterrows():
            subj = self._clean_text(row[self.subject_col])
            body = self._clean_text(row[self.body_col])
            
            # Explicit separator
            combined = f"SUBJECT: {subj} BODY: {body}"
            processed_texts.append(combined)
            
        # TfidfVectorizer expects a 1D array-like of strings
        return pd.Series(processed_texts)
