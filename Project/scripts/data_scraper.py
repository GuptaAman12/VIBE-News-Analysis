import feedparser
import pandas as pd
import os
import datetime

class NewsScraper:
    def __init__(self):
        # 1. Define RSS Sources (Indian Outlets)
        self.rss_sources = {
            'The Hindu': 'https://www.thehindu.com/news/national/feeder/default.rss',
            'Times of India': 'https://timesofindia.indiatimes.com/rssfeedstopstories.cms',
            'NDTV': 'https://feeds.feedburner.com/ndtvnews-top-stories',
            'India Today': 'https://www.indiatoday.in/rss/1206578',
            'News18': 'https://www.news18.com/rss/india.xml',
            'HindustanTimes': 'https://www.hindustantimes.com/feeds/rss/trending/rssfeed.xml',
            'FirstPost': 'https://www.firstpost.com/commonfeeds/v1/mfp/rss/web-stories.xml',
            'ABP News': 'https://news.abplive.com/home/feed'
        }
        self.df = pd.DataFrame()

    def fetch_data(self):
        print("--- 📡 Fetching RSS Feeds... ---")
        all_articles = []

        for source_name, url in self.rss_sources.items():
            print(f"Reading {source_name}...")
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    title = entry.get('title', '')
                    desc = entry.get('summary', '') if 'summary' in entry else entry.get('description', '')
                    published = entry.get('published', str(datetime.datetime.now()))

                    # Combine title and description
                    full_text = f"{title} {desc}"

                    all_articles.append({
                        'source': source_name,
                        'title': title,
                        'text': full_text,
                        'published': published
                    })
            except Exception as e:
                print(f"Error fetching {source_name}: {e}")

        self.df = pd.DataFrame(all_articles)
        print(f"Total articles fetched: {len(self.df)}")

        if self.df.empty:
            print("❌ No articles found.")

        return self.df

    def save_data(self, filepath):
        if self.df.empty:
            print("No data to save.")
            return

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if os.path.exists(filepath):
            print(f"Found existing data at {filepath}. Merging...")
            try:
                existing_df = pd.read_csv(filepath)
                # Concatenate
                combined_df = pd.concat([existing_df, self.df], ignore_index=True)
                # Drop duplicates based on title and source
                before_dedup = len(combined_df)
                combined_df.drop_duplicates(subset=['title', 'source'], keep='last', inplace=True)
                after_dedup = len(combined_df)
                print(f"Merged {len(self.df)} new articles. Total unique articles: {after_dedup}. (Dropped {before_dedup - after_dedup} duplicates)")
                combined_df.to_csv(filepath, index=False)
            except Exception as e:
                print(f"Error merging data: {e}")
        else:
            print(f"Creating new data file at {filepath}")
            self.df.to_csv(filepath, index=False)
            print(f"Saved {len(self.df)} articles.")

if __name__ == "__main__":
    scraper = NewsScraper()
    scraper.fetch_data()
    scraper.save_data(os.path.join('data', 'news_data.csv'))
