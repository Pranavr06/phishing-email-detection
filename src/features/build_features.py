import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts structural features related to URLs from the text.
    Separates real URLs from masked '[URL]' tokens to detect dataset artifacts.
    """
    def __init__(self, text_column='Body', remove_masked_urls=False):
        self.text_column = text_column
        self.remove_masked_urls = remove_masked_urls
        self.real_url_pattern = re.compile(
            r'(?:http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'
            r'|(?:www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+])+)',
            re.IGNORECASE
        )
        self.masked_url_pattern = re.compile(r'\[URL\]', re.IGNORECASE)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = pd.DataFrame(index=X.index)
        
        # Real URLs
        X_out['num_real_urls'] = X[self.text_column].apply(
            lambda text: len(self.real_url_pattern.findall(str(text))) if pd.notnull(text) else 0
        )
        
        # Masked URLs
        X_out['num_masked_urls'] = X[self.text_column].apply(
            lambda text: len(self.masked_url_pattern.findall(str(text))) if pd.notnull(text) else 0
        )
        
        # Total URLs
        X_out['num_urls'] = X_out['num_real_urls'] + X_out['num_masked_urls']
        
        # Unique URLs (only for real URLs as [URL] is just a token)
        X_out['num_unique_real_urls'] = X[self.text_column].apply(
            lambda text: len(set(self.real_url_pattern.findall(str(text)))) if pd.notnull(text) else 0
        )
        
        if self.remove_masked_urls:
            X_out = X_out.drop(columns=['num_masked_urls', 'num_urls'])
            
        return X_out

class HTMLFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts HTML-based features before HTML is stripped out.
    """
    def __init__(self, text_column='Body'):
        self.text_column = text_column
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.anchor_tag_pattern = re.compile(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = pd.DataFrame(index=X.index)
        
        X_out['num_html_tags'] = X[self.text_column].apply(
            lambda text: len(self.html_tag_pattern.findall(str(text))) if pd.notnull(text) else 0
        )
        X_out['has_html'] = (X_out['num_html_tags'] > 0).astype(int)
        
        X_out['num_anchor_tags'] = X[self.text_column].apply(
            lambda text: len(self.anchor_tag_pattern.findall(str(text))) if pd.notnull(text) else 0
        )
        return X_out

class PunctuationFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts targeted punctuation and casing features.
    """
    def __init__(self, text_column='Body'):
        self.text_column = text_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = pd.DataFrame(index=X.index)
        
        X_out['num_exclamation_marks'] = X[self.text_column].apply(
            lambda t: str(t).count('!') if pd.notnull(t) else 0
        )
        X_out['num_question_marks'] = X[self.text_column].apply(
            lambda t: str(t).count('?') if pd.notnull(t) else 0
        )
        X_out['num_currency_symbols'] = X[self.text_column].apply(
            lambda t: sum(str(t).count(c) for c in ['$','€','£','¥']) if pd.notnull(t) else 0
        )
        
        def uppercase_ratio(text):
            if pd.isnull(text): return 0.0
            text_str = str(text)
            alphas = sum(1 for c in text_str if c.isalpha())
            if alphas == 0: return 0.0
            uppers = sum(1 for c in text_str if c.isupper())
            return uppers / alphas

        X_out['uppercase_ratio'] = X[self.text_column].apply(uppercase_ratio)
        return X_out

class UrgencyFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts urgency indicators using a documented phrase set.
    Inspects both Subject and Body.
    """
    def __init__(self, subject_col='Subject', body_col='Body'):
        self.subject_col = subject_col
        self.body_col = body_col
        self.urgency_phrases = [
            'urgent', 'immediately', 'action required', 'verify now',
            'account suspended', 'account locked', 'security alert',
            'unusual activity', 'confirm your account', 'limited time'
        ]
        self.pattern = re.compile(r'\b(?:' + '|'.join(self.urgency_phrases) + r')\b', re.IGNORECASE)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = pd.DataFrame(index=X.index)
        
        def count_urgency(row):
            subj = str(row[self.subject_col]) if pd.notnull(row[self.subject_col]) else ""
            body = str(row[self.body_col]) if pd.notnull(row[self.body_col]) else ""
            combined = subj + " " + body
            return len(self.pattern.findall(combined))

        X_out['urgency_keyword_count'] = X.apply(count_urgency, axis=1)
        X_out['has_urgency'] = (X_out['urgency_keyword_count'] > 0).astype(int)
        return X_out
