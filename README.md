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
-   `VIBE_Headline_Frame_Game_Final.ipynb`: Labels headlines by framing and trains a TF-IDF + LogisticRegression classifier (with an optional BERTopic cell).
-   `Project/scripts/`: Modular Python scripts for the pipeline.
    -   `data_scraper.py`: Scrapes and merges data.
    -   `data_processor.py`: Cleans and analyzes data.
    -   `visualizer.py`: Generates plots.
    -   `main.py`: Orchestrates the entire workflow.
    -   `topics.py`: Whole-word keyword topic assignment.
    -   `project_paths.py`: Shared data/results paths that work from any directory.
    -   `test_project.py`: Assert-based self-check.
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
    pip install -r requirements.txt
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

## Checks

Run the assert-based self-check (no network needed):

```bash
python Project/scripts/test_project.py
```

## Results

A run of the pipeline produced the following snapshot: **45 headlines** from 6 Indian outlets (ABP News, FirstPost, HindustanTimes, India Today, NDTV, Times of India) across 4 topics, filtered from 358 scraped articles.

-   **Topic mix**: Politics/Elections (15), Environment/Health (15), Economy/Market (8), Crime/Law (7).
-   **Sentiment** (VADER compound, -1 to +1): slightly positive overall (+0.15). FirstPost was the most positive (+0.48) and HindustanTimes the most negative (-0.06). By topic, Crime/Law was the most negative (-0.32) and Economy/Market the most positive (+0.39).
-   **Subjectivity** (TextBlob, 0 = fact, 1 = opinion): Times of India (0.48) and HindustanTimes (0.43) were the most opinionated, while India Today (0.06) and ABP News (0.11) were the most factual. Economy/Market headlines were the most opinionated topic (0.42).
-   **Named entities**: 150 mentions detected; the most-mentioned were India, Putin and RBI (the saved NER file keeps the 15 most frequent, 68 rows).

Generated visualizations in `Project/results/`:

-   `plot_sentiment_by_topic.png` — average sentiment by topic and outlet
-   `plot_subjectivity_bias.png` — subjectivity distribution per outlet
-   `plot_bias_heatmap.png` — sentiment heatmap (outlet × topic)
-   `plot_readability_by_source.png` — Flesch reading ease by outlet
-   `plot_ner_sentiment_by_source.png` — sentiment for top entities by outlet
-   `wordcloud_<topic>.png` — word clouds for each topic

> These figures come from a single small snapshot and topic assignment uses simple keyword matching, so treat them as illustrative rather than representative. Re-run the pipeline to refresh them.
