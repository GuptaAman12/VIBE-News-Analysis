import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

class NewsVisualizer:
    def __init__(self):
        sns.set_theme(style="whitegrid")
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)

    def load_data(self, processed_filepath, ner_filepath):
        try:
            self.df = pd.read_csv(processed_filepath)
            print(f"Loaded {len(self.df)} processed articles.")
        except FileNotFoundError:
            print(f"File not found: {processed_filepath}")
            self.df = pd.DataFrame()

        try:
            self.ner_df = pd.read_csv(ner_filepath)
            print(f"Loaded {len(self.ner_df)} NER entries.")
        except FileNotFoundError:
            print(f"File not found: {ner_filepath}")
            self.ner_df = pd.DataFrame()

    def generate_plots(self):
        if self.df.empty:
            print("No data to visualize.")
            return

        print("\n--- 📊 Generating Plots... ---")

        # PLOT 1: Sentiment Distribution by Source and Topic
        if 'sentiment_score' in self.df.columns:
            plt.figure(figsize=(14, 7))
            sns.barplot(data=self.df, x='topic', y='sentiment_score', hue='source', errorbar=None, palette='viridis')
            plt.title('Average Sentiment Score by Topic and News Outlet', fontsize=16)
            plt.axhline(0, color='black', linewidth=1)
            plt.ylabel('Sentiment (Negative < 0 < Positive)')
            plt.xticks(rotation=15)
            plt.legend(title='News Source')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'plot_sentiment_by_topic.png'))
            print("Saved plot: plot_sentiment_by_topic.png")
            plt.close()

        # PLOT 2: Subjectivity Analysis (Fact vs Opinion)
        if 'subjectivity_score' in self.df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=self.df, x='source', y='subjectivity_score', palette='coolwarm')
            plt.title('Subjectivity in Headlines (0=Fact, 1=Opinion)', fontsize=16)
            plt.ylabel('Subjectivity Score')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'plot_subjectivity_bias.png'))
            print("Saved plot: plot_subjectivity_bias.png")
            plt.close()

        # PLOT 3: Sentiment Heatmap
        if 'sentiment_score' in self.df.columns:
            try:
                pivot_df = self.df.pivot_table(index='source', columns='topic', values='sentiment_score', aggfunc='mean')
                plt.figure(figsize=(10, 6))
                sns.heatmap(pivot_df, annot=True, cmap='RdBu', center=0, linewidths=1, linecolor='black')
                plt.title('Media Bias Heatmap: Sentiment Intensity', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'plot_bias_heatmap.png'))
                print("Saved plot: plot_bias_heatmap.png")
                plt.close()
            except Exception as e:
                print(f"Could not generate heatmap: {e}")

        # PLOT 4: Readability Score by Outlet
        if 'readability_score' in self.df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=self.df, x='source', y='readability_score', palette='pastel')
            plt.title('Headline Readability by News Outlet (Higher = Easier to Read)', fontsize=16)
            plt.ylabel('Flesch Reading Ease Score')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'plot_readability_by_source.png'))
            print("Saved plot: plot_readability_by_source.png")
            plt.close()

        # PLOT 5: Word Clouds per Topic
        print("Generating Word Clouds...")
        unique_topics = self.df['topic'].unique()
        for topic in unique_topics:
            topic_text = " ".join(self.df[self.df['topic'] == topic]['cleaned_text'].astype(str))
            if len(topic_text) > 10:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(topic_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Word Cloud: {topic}', fontsize=15)
                filename = f'wordcloud_{topic.replace("/", "_")}.png'
                plt.savefig(os.path.join(self.results_dir, filename))
                print(f"Saved plot: {filename}")
                plt.close()

        # PLOT 6: Sentiment by Top Mentioned Entity
        if not self.ner_df.empty:
            # Calculate average sentiment for each entity by source
            entity_plot_df = self.ner_df.groupby(['entity', 'source'])['sentiment_score'].mean().reset_index()

            plt.figure(figsize=(15, 8))
            sns.barplot(data=entity_plot_df, x='entity', y='sentiment_score', hue='source', palette='muted', dodge=True)
            plt.title('Average Sentiment for Top 15 Entities by News Outlet', fontsize=16)
            plt.ylabel('Sentiment (Negative < 0 < Positive)')
            plt.axhline(0, color='black', linewidth=1, linestyle='--')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='News Source', loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'plot_ner_sentiment_by_source.png'))
            print("Saved plot: plot_ner_sentiment_by_source.png")
            plt.close()

if __name__ == "__main__":
    visualizer = NewsVisualizer()
    visualizer.load_data(os.path.join('data', 'processed_news_data.csv'), os.path.join('data', 'ner_data.csv'))
    visualizer.generate_plots()
