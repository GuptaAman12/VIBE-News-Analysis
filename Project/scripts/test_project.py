"""Assert-based self-check: python Project/scripts/test_project.py"""
import os

from project_paths import DATA_DIR, RESULTS_DIR, NLTK_DIR, PROJECT_DIR
from topics import assign_topic

assert os.path.basename(PROJECT_DIR) == 'Project'
assert os.path.dirname(DATA_DIR) == PROJECT_DIR
assert os.path.dirname(RESULTS_DIR) == PROJECT_DIR
assert os.path.basename(NLTK_DIR) == 'nltk_data'

assert assign_topic('Pollution levels rise in Delhi') == 'Environment/Health'
assert assign_topic('Election poll results announced') == 'Politics/Elections'
assert assign_topic('Staff training announced') == 'Other'
assert assign_topic('The supreme court gave its ruling') == 'Crime/Law'
assert assign_topic(None) == 'Other'

print('All checks passed.')
