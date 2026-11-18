"""Shared paths so every script writes inside the project, whatever the cwd."""
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
REPO_DIR = os.path.dirname(PROJECT_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
NLTK_DIR = os.path.join(REPO_DIR, 'nltk_data')
