"""Keyword topics used to filter headlines."""
import re

TOPICS = {
    'Politics/Elections': ['election', 'poll', 'bjp', 'congress', 'vote', 'modi', 'gandhi'],
    'Economy/Market': ['sensex', 'nifty', 'economy', 'inflation', 'rbi', 'gdp', 'bank'],
    'Environment/Health': ['pollution', 'smog', 'aqi', 'health', 'hospital', 'doctor', 'rain'],
    'Crime/Law': ['police', 'court', 'supreme court', 'arrest', 'case', 'justice'],
}


def assign_topic(text):
    """First topic with a whole-word keyword match, else 'Other'."""
    if not isinstance(text, str):
        return 'Other'
    text_lower = text.lower()
    for topic, keywords in TOPICS.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                return topic
    return 'Other'
