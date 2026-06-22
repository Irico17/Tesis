"""Downloaders de datasets originales."""

from phishing_pipeline.downloaders.kaggle_phishing import download_kaggle_phishing
from phishing_pipeline.downloaders.phish_mmf import download_phish_mmf
from phishing_pipeline.downloaders.spam_genuine import download_spam_genuine

__all__ = [
    "download_kaggle_phishing",
    "download_spam_genuine",
    "download_phish_mmf",
]
