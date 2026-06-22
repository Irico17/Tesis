"""Cleaners y mappers al esquema canónico."""

from phishing_pipeline.cleaners.kaggle_phishing import clean_kaggle_phishing
from phishing_pipeline.cleaners.phish_mmf import clean_phish_mmf, parse_phish_mmf_jsonl
from phishing_pipeline.cleaners.spam_genuine import clean_spam_genuine

__all__ = [
    "clean_kaggle_phishing",
    "clean_spam_genuine",
    "clean_phish_mmf",
    "parse_phish_mmf_jsonl",
]
