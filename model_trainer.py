import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class YouTubePerformancePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare all features from raw data"""
        from thumbnail_analyzer import ThumbnailAnalyzer
        from text_analyzer import TextAnalyzer
        
        thumbnail_analyzer = ThumbnailAnalyzer()
        text_analyzer = TextAnalyzer()
        
        features_list = []
        
        for idx, row in df.iterrows():
            features = {}
            
            # Analyze thumbnail
            thumb_features = thumbnail_analyzer.analyze_thumbnail(row['thumbnail_url'])
            features.update(thumb_features)
            
            # Analyze title
            title_features = text_analyzer.analyze_title(row['title'])
            features.update(title_features)
            
            # Analyze description
            desc_features = text_analyzer.analyze_description(row['description'])
            features.update(desc_features)
            
            # Analyze tags
            tag_features = text_analyzer.analyze_tags(row['tags'])
            features.update(tag_features)
            
            # Add video metadata
            features['category_id'] = row.get('category_id', 0)
            features['duration_seconds'] = self.parse_duration(row.get('duration', 'PT0M0S'))
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        self.feature_columns = features_df.columns.tolist()
        
        return features_df
    
    def parse_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration to seconds"""
        import re
        match = re.match(r'PT(\d+H)?(\d+M)?(\d+S)?', duration_str)
        if not match:
            return 0
        
        hours = int(match.group(1)[:-1]) if match.group(1) else 0
        minutes = int(match.group(2)[:-1]) if match.group(2) else 0
        seconds = int(match.group(3)[:-1]) if match.group(3) else 0
        
        return hours * 3600 + minutes * 60 + seconds
    
    def train_models(self, features_df: pd.DataFrame, targets_df: pd.DataFrame):
        """Train models for different prediction targets"""
        
        # Define targets
        targets = {
            'views_48h': 'estimated_48h_views',
            'engagement_rate': 'ctr_proxy'
        }
        
        for model_name, target_col in targets.items():
            print(f"\nTraining model for: {model_name}")
            
            # Prepare data
            X = features_df.fillna(0).values
            y = targets_df[target_col].values
            
            # Remove outliers
            Q1 = np.percentile(y, 25)
            Q3 = np.percentile(y, 75)
            IQR = Q3 - Q1
            mask = (y >= Q1 - 1.5 * IQR) & (y <= Q3 + 1.5 * IQR)
            X = X[mask]
            y = y[mask]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[model_name] = scaler
            
            # Train model
            if model_name == 'views_48h':
                # For views prediction, use Gradient Boosting
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            else:
                # For engagement rate, use Random Forest
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            
            model.fit(X_train_scaled, y_train)
            self.models[model_name] = model
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"MAE: {mae:.2f}")
            print(f"R² Score: {r2:.2f}")
    
    def predict(self, features: dict) -> dict:
        """Make predictions for new video"""
        predictions = {}
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([features])
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reorder columns
        features_df = features_df[self.feature_columns]
        
        for model_name, model in self.models.items():
            scaler = self.scalers[model_name]
            X_scaled = scaler.transform(features_df.values)
            prediction = model.predict(X_scaled)[0]
            predictions[model_name] = prediction
        
        # Calculate overall score (1-100)
        views_score = min(predictions['views_48h'] / 10000 * 50, 50)
        engagement_score = min(predictions['engagement_rate'] * 500, 50)
        overall_score = min(views_score + engagement_score, 100)
        
        predictions['overall_score'] = overall_score
        
        return predictions
    
    def get_feature_importance(self, model_name: str = 'views_48h') -> pd.DataFrame:
        """Get feature importance for interpretation"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        importance = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, path: str = 'models/'):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self.models, f'{path}/models.pkl')
        joblib.dump(self.scalers, f'{path}/scalers.pkl')
        joblib.dump(self.feature_columns, f'{path}/feature_columns.pkl')
        
        print(f"Models saved to {path}")
    
    def load_model(self, path: str = 'models/'):
        """Load trained models"""
        self.models = joblib.load(f'{path}/models.pkl')
        self.scalers = joblib.load(f'{path}/scalers.pkl')
        self.feature_columns = joblib.load(f'{path}/feature_columns.pkl')
        
        print("Models loaded successfully")
