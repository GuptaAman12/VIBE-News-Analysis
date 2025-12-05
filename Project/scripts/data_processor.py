import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from collections import Counter
import textstat
import os

class NewsProcessor:
    def __init__(self):
        # 2. Define Recent Events (Keywords)
        self.topics = {
            'Politics/Elections': ['election', 'poll', 'bjp', 'congress', 'vote', 'modi', 'gandhi'],
            'Economy/Market': ['sensex', 'nifty', 'economy', 'inflation', 'rbi', 'gdp', 'bank'],
            'Environment/Health': ['pollution', 'smog', 'aqi', 'health', 'hospital', 'doctor', 'rain'],
            'Crime/Law': ['police', 'court', 'supreme court', 'arrest', 'case', 'justice']
        }
        
        # Setup NLTK
        print("Downloading NLTK resources...")
        # Set local NLTK path
        nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        
        try:
            nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
            nltk.download('vader_lexicon', download_dir=nltk_data_dir, quiet=True)
            nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
            nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
        except Exception as e:
            print(f"Error downloading NLTK resources: {e}")
        
        # Setup Spacy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except IOError:
            print("Error: 'en_core_web_sm' model not found. Please run: python -m spacy download en_core_web_sm")
            self.nlp = None

    def load_data(self, filepath):
        try:
            self.df = pd.read_csv(filepath)
            print(f"Loaded {len(self.df)} articles from {filepath}")
            return self.df
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            self.df = pd.DataFrame()
            return self.df

    def filter_topics_and_clean(self):
        print("\n--- 🧹 Filtering Topics & Cleaning Text... ---")

        if self.df.empty:
            print("No data to process.")
            return

        # 1. Assign Topics
        def assign_topic(text):
            if not isinstance(text, str): return "Other"
            text_lower = text.lower()
            for topic, keywords in self.topics.items():
                for k in keywords:
                    if k in text_lower:
                        return topic
            return "Other"

        self.df['topic'] = self.df['text'].apply(assign_topic)

        # Filter out 'Other'
        self.df = self.df[self.df['topic'] != 'Other'].copy()
        print(f"Articles remaining after topic filtering: {len(self.df)}")

        # 2. Clean Text
        stop_words = set(stopwords.words('english'))
        def clean_text(text):
            if not isinstance(text, str): return ""
            text = text.lower()
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            words = text.split()
            words = [w for w in words if w not in stop_words]
            return " ".join(words)

        self.df['cleaned_text'] = self.df['text'].apply(clean_text)

    def analyze_bias(self):
        print("\n--- 🧠 Analyzing Sentiment & Subjectivity... ---")
        if self.df.empty: return

        sid = SentimentIntensityAnalyzer()

        def get_metrics(text):
            if not isinstance(text, str): return pd.Series([0.0, 0.0])
            # VADER for Sentiment
            polarity = sid.polarity_scores(text)['compound']
            # TextBlob for Subjectivity
            subjectivity = TextBlob(text).sentiment.subjectivity
            return pd.Series([polarity, subjectivity])

        self.df[['sentiment_score', 'subjectivity_score']] = self.df['text'].apply(get_metrics)

    def analyze_ner(self):
        print("\n--- 🔬 Analyzing Named Entities (NER)... ---")
        if self.df.empty or self.nlp is None: 
            self.top_entity_df = pd.DataFrame()
            return

        entity_labels = ['PERSON', 'ORG', 'GPE']
        
        entity_data = []
        for index, row in self.df.iterrows():
            if not isinstance(row['text'], str): continue
            doc = self.nlp(row['text'])
            for ent in doc.ents:
                if ent.label_ in entity_labels:
                    entity_data.append({
                        'entity': ent.text.strip(),
                        'label': ent.label_,
                        'source': row['source'],
                        'topic': row['topic'],
                        'sentiment_score': row['sentiment_score']
                    })
        
        self.entity_df = pd.DataFrame(entity_data)
        if not self.entity_df.empty:
             # Filter top entities
            common_entities = [ent for ent, count in Counter(self.entity_df['entity']).most_common(15)]
            self.top_entity_df = self.entity_df[self.entity_df['entity'].isin(common_entities)]
            print(f"Found {len(self.entity_df)} entities. Filtered to top 15 frequent ones.")
        else:
            self.top_entity_df = pd.DataFrame()

    def analyze_readability(self):
        print("\n--- 📖 Analyzing Readability Scores... ---")
        if self.df.empty: return

        def get_readability(text):
             if not isinstance(text, str): return 0
             return textstat.flesch_reading_ease(text)

        self.df['readability_score'] = self.df['text'].apply(get_readability)

    def save_processed_data(self, main_filepath, ner_filepath):
        # Ensure directory exists
        os.makedirs(os.path.dirname(main_filepath), exist_ok=True)
        
        if not self.df.empty:
            self.df.to_csv(main_filepath, index=False)
            print(f"Saved processed data to {main_filepath}")
        
        if hasattr(self, 'top_entity_df') and not self.top_entity_df.empty:
            self.top_entity_df.to_csv(ner_filepath, index=False)
            print(f"Saved NER data to {ner_filepath}")

if __name__ == "__main__":
    processor = NewsProcessor()
    df = processor.load_data(os.path.join('data', 'news_data.csv'))
    if not df.empty:
        processor.filter_topics_and_clean()
        processor.analyze_bias()
        processor.analyze_readability()
        processor.analyze_ner()
        processor.save_processed_data(os.path.join('data', 'processed_news_data.csv'), os.path.join('data', 'ner_data.csv'))
