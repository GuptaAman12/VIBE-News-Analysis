# ==============================================================================
# Final, Robust Python Script for VS Code
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import nltk
import re
from newsapi import NewsApiClient
import os
from datetime import datetime, timedelta

# ==============================================================================
# Section 1: Dataset Collection
# ==============================================================================
def collect_data():
    print("--- Starting Section 1: Dataset Collection ---")
    # --- Configuration ---
    api_key = '54e4b371f0e041d9bab6c34a8bb72c12' # Your key is already here
    
    newsapi = NewsApiClient(api_key=api_key)

    news_sources = {
        'The Hindu': 'thehindu.com',
        'Times of India': 'timesofindia.indiatimes.com',
        'The Wire': 'thewire.in',
        'Swarajya': 'swarajyamag.com'
    }

    # --- Event and Time Window (UPDATED QUERY) ---
    # The query 'India' was too broad. Let's use a more specific but
    # high-frequency topic like 'politics'.
    # If this doesn't work, other good options are: 'economy', 'market', 'government'
    query = 'politics'
    
    # This date logic is correct. It searches a completed 48-hour window.
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    three_days_ago = today - timedelta(days=3)
    to_date = yesterday.strftime('%Y-%m-%d')
    from_date = three_days_ago.strftime('%Y-%m-%d')

    print(f"Searching for topic '{query}' from {from_date} to {to_date}.")

    all_articles = []
    for source_name, source_domain in news_sources.items():
        print(f"Fetching articles from {source_name}...")
        try:
            # We've changed sort_by to 'publishedAt' for more consistent results.
            articles = newsapi.get_everything(q=query,
                                              domains=source_domain,
                                              from_param=from_date,
                                              to=to_date,
                                              language='en',
                                              sort_by='publishedAt', 
                                              page_size=100)
            for article in articles['articles']:
                all_articles.append({ 'source': source_name, 'title': article['title'], 'description': article['description'], 'publishedAt': article['publishedAt'], 'url': article['url'] })
        except Exception as e:
            print(f"Could not fetch articles from {source_name}. Error: {e}")

    df = pd.DataFrame(all_articles)
    if not df.empty:
        df.to_csv('raw_headlines.csv', index=False)
        print(f"\nCollected {len(df)} articles and saved to raw_headlines.csv")
    return df

# ==============================================================================
# Section 2: Pre-processing
# ==============================================================================
def preprocess_data(df):
    print("\n--- Starting Section 2: Pre-processing ---")
    nlp = spacy.load('en_core_web_sm')
    stop_words = set(nltk.corpus.stopwords.words('english'))

    def clean_text(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return text

    def remove_stopwords(text):
        tokens = nltk.word_tokenize(text)
        return ' '.join([word for word in tokens if word not in stop_words])

    def lemmatize_text(text):
        doc = nlp(text)
        return ' '.join([token.lemma_ for token in doc])

    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df['cleaned_text'] = df['text'].apply(clean_text).apply(remove_stopwords)
    df['lemmatized_text'] = df['cleaned_text'].apply(lemmatize_text)
    
    df.drop_duplicates(subset=['title', 'source'], inplace=True)
    df = df[df['lemmatized_text'].str.strip() != '']

    df.to_csv('clean_headlines.csv', index=False)
    print(f"Cleaned data has {len(df)} articles and is saved to clean_headlines.csv")
    return df

# ==============================================================================
# Section 3: Analysis
# ==============================================================================
def analyze_data(df):
    print("\n--- Starting Section 3: Analysis ---")
    news_sources = df['source'].unique()
    
    # --- Sentiment Analysis ---
    sid = SentimentIntensityAnalyzer()
    df['sentiment_scores'] = df['text'].apply(lambda text: sid.polarity_scores(text))
    df = pd.concat([df.drop(['sentiment_scores'], axis=1), df['sentiment_scores'].apply(pd.Series)], axis=1)
    
    avg_sentiment = df.groupby('source')[['neg', 'neu', 'pos', 'compound']].mean().reset_index()
    print("\n--- Average Sentiment per Outlet ---")
    print(avg_sentiment.to_string())

    return df, avg_sentiment

# ==============================================================================
# Section 4: Visualization
# ==============================================================================
def create_visualizations(df, avg_sentiment):
    print("\n--- Starting Section 4: Visualization ---")
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    news_sources = df['source'].unique()

    # 1. Word Clouds
    for source in news_sources:
        text = ' '.join(df[df['source'] == source]['lemmatized_text'])
        if text:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud for {source}')
            plt.savefig(f'plots/word_cloud_{source.replace(" ", "_")}.png')
            print(f"Saved word cloud for {source}.")
            plt.close()

# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    # Step 1: Collect Data
    raw_df = collect_data()
    
    # Step 2: Check if data was collected BEFORE proceeding
    if raw_df is not None and not raw_df.empty:
        
        # Step 3: Pre-process Data
        clean_df = preprocess_data(raw_df)
        
        # Step 4: Analyze Data
        analyzed_df, avg_sentiment = analyze_data(clean_df)
        
        # Step 5: Create Visualizations
        create_visualizations(analyzed_df, avg_sentiment)
        
        print("\n✅ Project execution completed successfully!")
    else:
        print("\n❌ Project execution stopped because no articles were found. Please try a different search query.")