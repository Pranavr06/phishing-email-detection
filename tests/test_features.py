import pytest
import pandas as pd
import numpy as np
from src.features.build_features import (
    URLFeatureExtractor,
    HTMLFeatureExtractor,
    PunctuationFeatureExtractor,
    UrgencyFeatureExtractor
)
from src.data.preprocess import TextPreprocessor

def test_url_extractor():
    data = pd.DataFrame({
        'Body': [
            "Visit http://example.com and https://test.com",
            "Go to www.google.com or [URL]",
            "Check example.com", # We didn't explicitly match bare without www/http in the regex to avoid false positives, but let's test what it does.
            "Duplicate http://test.com and http://test.com",
            np.nan,
            ""
        ]
    })
    extractor = URLFeatureExtractor()
    out = extractor.transform(data)
    
    assert out.loc[0, 'num_urls'] == 2
    assert out.loc[1, 'num_urls'] == 2
    assert out.loc[3, 'num_urls'] == 2
    assert out.loc[3, 'num_unique_urls'] == 1
    assert out.loc[4, 'num_urls'] == 0

def test_html_extractor():
    data = pd.DataFrame({
        'Body': [
            "Plain text email.",
            "<b>HTML</b> email with <a href='http://test.com'>link</a>.",
            np.nan
        ]
    })
    extractor = HTMLFeatureExtractor()
    out = extractor.transform(data)
    
    assert out.loc[0, 'num_html_tags'] == 0
    assert out.loc[1, 'num_html_tags'] == 3 # <b>, </b>, <a>
    assert out.loc[1, 'num_anchor_tags'] == 1
    assert out.loc[1, 'has_html'] == 1
    assert out.loc[2, 'num_html_tags'] == 0

def test_punctuation_extractor():
    data = pd.DataFrame({
        'Body': [
            "URGENT!!! Please pay $100 or €50.",
            "Are you sure???",
            np.nan
        ]
    })
    extractor = PunctuationFeatureExtractor()
    out = extractor.transform(data)
    
    assert out.loc[0, 'num_exclamation_marks'] == 3
    assert out.loc[0, 'num_currency_symbols'] == 2
    assert out.loc[1, 'num_question_marks'] == 3
    
    # Uppercase ratio test: "URGENT Please pay or " -> 6 upper / 19 letters = ~0.31
    ratio = out.loc[0, 'uppercase_ratio']
    assert 0.2 < ratio < 0.4
    assert out.loc[2, 'uppercase_ratio'] == 0.0

def test_urgency_extractor():
    data = pd.DataFrame({
        'Subject': ["Action Required", "Hello", np.nan],
        'Body': ["Verify now or account locked", "Just a test", "Urgent"]
    })
    extractor = UrgencyFeatureExtractor()
    out = extractor.transform(data)
    
    assert out.loc[0, 'urgency_keyword_count'] == 3 # Action Required, Verify now, account locked
    assert out.loc[1, 'urgency_keyword_count'] == 0
    assert out.loc[2, 'urgency_keyword_count'] == 1
