# VIBE News Analysis

This project analyzes news headlines from various Indian news outlets to detect bias, sentiment, and framing. It scrapes RSS feeds, processes the text using NLP techniques, and generates visualizations.

## Features

-   **Incremental Scraping**: Fetches news from RSS feeds and merges it with existing data, avoiding duplicates.
-   **Sentiment Analysis**: Uses VADER and TextBlob to analyze sentiment and subjectivity.
-   **Named Entity Recognition (NER)**: Identifies top entities (people, organizations) mentioned in the news.
-   **Readability Analysis**: Calculates Flesch Reading Ease scores.
-   **Visualizations**: Generates word clouds, heatmaps, and bar charts to visualize the analysis.

## Project Structure

-   `VIBE_new.ipynb`: The original Jupyter Notebook with exploratory analysis.
-   `Project/scripts/`: Modular Python scripts for the pipeline.
    -   `data_scraper.py`: Scrapes and merges data.
    -   `data_processor.py`: Cleans and analyzes data.
    -   `visualizer.py`: Generates plots.
    -   `main.py`: Orchestrates the entire workflow.
-   `Project/data/`: Stores the CSV data files (not uploaded to git).
-   `Project/results/`: Stores the generated plots (not uploaded to git).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/GuptaAman12/VIBE-News-Analysis.git
    cd VIBE-News-Analysis
    ```

2.  Install the required Python packages:
    ```bash
    pip install feedparser pandas numpy matplotlib seaborn textblob nltk wordcloud spacy textstat
    ```

3.  Download the spaCy English model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

To run the full analysis pipeline (scrape -> process -> visualize):

```bash
python Project/scripts/main.py
```

This will:
1.  Fetch the latest news from the configured RSS feeds.
2.  Merge it with `Project/data/news_data.csv`.
3.  Process the data and save it to `Project/data/processed_news_data.csv`.
4.  Generate visualizations in the `Project/results/` directory.
