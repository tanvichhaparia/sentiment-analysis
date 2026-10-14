import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import preprocess_dataframe
from src.trend_extraction import extract_trends_from_phrases
from src.sentiment_model import SentimentModel

def main():
    input_path = "data/raw/synthetic_fashion_data.csv"
    output_path = "data/processed/processed_fashion_data.csv"
    model_path = "models/sentiment_model/sentiment_pipeline.pkl"
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return
        
    print("Loading data...")
    df = pd.read_csv(input_path)
    
    print("Running NLP preprocessing (this may take a minute)...")
    df = preprocess_dataframe(df, text_column='text')
    
    print("Extracting fashion trends...")
    df['detected_trends'] = extract_trends_from_phrases(df['noun_phrases'])
    
    # 3. Unsupervised Topic Discovery (NMF)
    print("Running Unsupervised Topic Discovery (NMF)...")
    from src.topic_discovery import run_topic_discovery, save_topics
    topic_res = run_topic_discovery(df, text_column='clean_text', n_topics=8, method='nmf')
    df['discovered_topic'] = topic_res['dominant_topics']
    save_topics(topic_res, output_path='reports/discovered_topics.json')

    # Predict sentiments using our trained model if it exists
    if os.path.exists(model_path):
        print("Predicting sentiment with trained model...")
        model = SentimentModel()
        model.load(model_path)
        df['predicted_sentiment'] = model.predict(df['text'])
    else:
        print("Trained model not found, using true labels for predicted sentiment (run train_model.py first).")
        df['predicted_sentiment'] = df['sentiment_label']
        
    # Ensure date is string for saving
    df['date'] = df['date'].astype(str)
    
    # Save processed data
    # Convert list column to string so it saves to CSV nicely, or just save to Parquet. 
    # CSV is easier for the user to view.
    df['detected_trends'] = df['detected_trends'].apply(lambda x: ",".join(x))
    df['noun_phrases'] = df['noun_phrases'].apply(lambda x: ",".join(x))
    
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    main()
