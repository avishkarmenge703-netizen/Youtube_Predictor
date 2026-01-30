import pandas as pd
import numpy as np
from youtube_api import YouTubeDataCollector
from text_analyzer import TextAnalyzer
from thumbnail_analyzer import ThumbnailAnalyzer
from model_trainer import YouTubePerformancePredictor
import joblib
import os

def collect_training_data(api_key, niches, videos_per_niche=50):
    """Collect training data from YouTube"""
    collector = YouTubeDataCollector(api_key)
    text_analyzer = TextAnalyzer()
    thumbnail_analyzer = ThumbnailAnalyzer()
    
    all_data = []
    
    for niche in niches:
        print(f"Collecting data for niche: {niche}")
        
        # Get videos from YouTube
        videos_df = collector.search_videos(
            query=niche,
            max_results=videos_per_niche,
            days_back=60
        )
        
        if len(videos_df) == 0:
            print(f"No videos found for {niche}")
            continue
        
        # Analyze each video
        for idx, row in videos_df.iterrows():
            try:
                print(f"Analyzing video {idx+1}/{len(videos_df)}: {row['title'][:50]}...")
                
                # Analyze text
                title_features = text_analyzer.analyze_title(row['title'])
                desc_features = text_analyzer.analyze_description(row['description'])
                tag_features = text_analyzer.analyze_tags(row.get('tags', []))
                
                # Analyze thumbnail
                thumbnail_features = thumbnail_analyzer.analyze_thumbnail(row['thumbnail_url'])
                
                if thumbnail_features is None:
                    continue
                
                # Combine data
                video_data = {
                    'video_id': row['video_id'],
                    'title': row['title'],
                    'title_features': title_features,
                    'thumbnail_features': thumbnail_features,
                    'desc_features': desc_features,
                    'tag_features': tag_features,
                    '48h_views': row['48h_views'],
                    '48h_likes': row['48h_likes'],
                    '48h_comments': row['48h_comments'],
                    'niche': niche
                }
                
                all_data.append(video_data)
                
            except Exception as e:
                print(f"Error analyzing video: {e}")
                continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save raw data
    df.to_pickle('training_data.pkl')
    print(f"Collected {len(df)} videos for training")
    
    return df

def train_model(df):
    """Train the prediction model"""
    predictor = YouTubePerformancePredictor()
    
    # Prepare features
    X = predictor.prepare_features(df)
    
    # Prepare targets
    y_views = df['48h_views'].values
    y_likes = df['48h_likes'].values
    y_comments = df['48h_comments'].values
    
    # Remove NaN values
    mask = ~np.isnan(y_views) & ~np.isnan(y_likes) & ~np.isnan(y_comments)
    X = X[mask]
    y_views = y_views[mask]
    y_likes = y_likes[mask]
    y_comments = y_comments[mask]
    
    print(f"Training on {len(X)} samples with {X.shape[1]} features")
    
    # Train models
    cv_scores = predictor.train_models(X, y_views, y_likes, y_comments)
    
    print("\nCross-validation R² Scores:")
    for target, score in cv_scores.items():
        print(f"  {target}: {score:.3f}")
    
    # Save models
    predictor.save_models()
    
    # Show feature importance
    print("\nTop 10 Features for View Prediction:")
    print(predictor.feature_importance['views'].head(10))
    
    return predictor

if __name__ == "__main__":
    # YouTube API Key (get from Google Cloud Console)
    API_KEY = "AIzaSyAFNFFB1fjNrtSr_tnpiTiZalPV23d6-qA"
    
    # Define niches to collect data from
    niches = [
        "tech review",
        "cooking tutorial",
        "fitness workout",
        "gaming highlights",
        "personal finance tips",
        "productivity tips"
    ]
    
    # Collect data
    print("Starting data collection...")
    df = collect_training_data(API_KEY, niches, videos_per_niche=30)
    
    # Train model
    print("\nTraining model...")
    predictor = train_model(df)
    
    print("\nModel training complete! You can now run the Flask app.")
