# VIBE News Analysis Pipeline - Final Version
import os
from data_scraper import NewsScraper
from data_processor import NewsProcessor
from visualizer import NewsVisualizer
from project_paths import DATA_DIR, RESULTS_DIR

def main():
    print("=== VIBE News Analysis Pipeline ===")
    
    raw_data_path = os.path.join(DATA_DIR, 'news_data.csv')
    processed_data_path = os.path.join(DATA_DIR, 'processed_news_data.csv')
    ner_data_path = os.path.join(DATA_DIR, 'ner_data.csv')
    
    # 1. Scrape Data
    print("\n[Step 1] Scraping Data...")
    scraper = NewsScraper()
    scraper.fetch_data()
    scraper.save_data(raw_data_path)
    
    # 2. Process Data
    print("\n[Step 2] Processing Data...")
    processor = NewsProcessor()
    df = processor.load_data(raw_data_path)
    if not df.empty:
        processor.filter_topics_and_clean()
        processor.analyze_bias()
        processor.analyze_readability()
        processor.analyze_ner()
        processor.save_processed_data(processed_data_path, ner_data_path)
    else:
        print("Skipping processing as no data was loaded.")
        return

    # 3. Visualize Data
    print("\n[Step 3] Visualizing Data...")
    visualizer = NewsVisualizer()
    visualizer.load_data(processed_data_path, ner_data_path)
    visualizer.generate_plots()
    
    print("\n=== Pipeline Complete! ===")
    print(f"Check '{RESULTS_DIR}' for plots and '{DATA_DIR}' for data files.")

if __name__ == "__main__":
    main()
