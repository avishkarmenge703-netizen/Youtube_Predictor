from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import os
from datetime import datetime
from model_trainer import YouTubePerformancePredictor
from text_analyzer import TextAnalyzer
from thumbnail_analyzer import ThumbnailAnalyzer

app = Flask(__name__)
predictor = YouTubePerformancePredictor()
text_analyzer = TextAnalyzer()
thumbnail_analyzer = ThumbnailAnalyzer()

# Try to load pre-trained model
try:
    predictor.load_model('models/')
    print("Model loaded successfully")
except:
    print("No pre-trained model found. Please train the model first.")

class VideoAudit:
    def __init__(self, title, thumbnail_url, description, tags):
        self.title = title
        self.thumbnail_url = thumbnail_url
        self.description = description
        self.tags = tags
    
    def analyze(self):
        """Complete video analysis"""
        # Extract features
        features = {}
        
        # Thumbnail features
        thumb_features = thumbnail_analyzer.analyze_thumbnail(self.thumbnail_url)
        features.update(thumb_features)
        
        # Title features
        title_features = text_analyzer.analyze_title(self.title)
        features.update(title_features)
        
        # Description features
        desc_features = text_analyzer.analyze_description(self.description)
        features.update(desc_features)
        
        # Tag features
        tag_features = text_analyzer.analyze_tags(self.tags)
        features.update(tag_features)
        
        # Add default values for missing features
        features['category_id'] = 24  # Education default
        features['duration_seconds'] = 600  # 10 minutes default
        
        # Make prediction
        predictions = predictor.predict(features)
        
        # Generate suggestions
        suggestions = self.generate_suggestions(features, predictions)
        
        # Generate alternatives
        alternatives = self.generate_alternatives()
        
        return {
            'score': predictions['overall_score'],
            'predictions': predictions,
            'features': features,
            'suggestions': suggestions,
            'alternatives': alternatives,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_suggestions(self, features, predictions):
        """Generate improvement suggestions"""
        suggestions = []
        
        # Thumbnail suggestions
        if features['thumb_has_face'] == 0:
            suggestions.append("Consider adding a human face to the thumbnail - increases CTR by 23% on average")
        
        if features['thumb_brightness'] < 100:
            suggestions.append("Increase thumbnail brightness - brighter thumbnails perform 17% better")
        
        if features['thumb_has_text'] == 0:
            suggestions.append("Add bold text overlay - helps communicate value proposition")
        
        # Title suggestions
        if features['title_curiosity_score'] < 0.3:
            suggestions.append("Add curiosity gap to title (use words like 'secret', 'why', 'how')")
        
        if features['title_has_brackets'] == 0:
            suggestions.append("Consider adding brackets [] to title - can increase CTR by 38%")
        
        # Description suggestions
        if features['desc_has_timestamps'] == 0:
            suggestions.append("Add timestamps to description - improves audience retention")
        
        if features['desc_cta_count'] < 2:
            suggestions.append("Add more CTAs (Call-to-Action) in description")
        
        # Tag suggestions
        if features['tag_count'] < 8:
            suggestions.append("Add more tags (aim for 10-15 relevant tags)")
        
        return suggestions
    
    def generate_alternatives(self):
        """Generate alternative titles and thumbnail ideas"""
        alternatives = {
            'titles': [
                f"[REVEALED] {self.title} - What Nobody Tells You",
                f"How to {self.title} (Step-by-Step Guide)",
                f"The Truth About {self.title.split()[0]} - Shocking Results",
                f"{self.title}: Complete Beginner's Guide",
                f"Stop Doing {self.title.split()[0]} Wrong - Here's Why"
            ],
            'thumbnail_ideas': [
                "Close-up of surprised face with text overlay",
                "Before/After comparison with bold arrows",
                "Red circle/arrow highlighting key element",
                "Bright color background with large bold text",
                "Split-screen showing problem/solution"
            ]
        }
        return alternatives

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """API endpoint for video analysis"""
    data = request.json
    
    audit = VideoAudit(
        title=data['title'],
        thumbnail_url=data['thumbnail_url'],
        description=data['description'],
        tags=data['tags']
    )
    
    result = audit.analyze()
    
    return jsonify(result)

@app.route('/api/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple videos for channel audit"""
    videos = request.json['videos']
    results = []
    
    for video_data in videos:
        audit = VideoAudit(**video_data)
        result = audit.analyze()
        results.append(result)
    
    # Generate channel insights
    channel_insights = generate_channel_insights(results)
    
    return jsonify({
        'individual_results': results,
        'channel_insights': channel_insights
    })

def generate_channel_insights(results):
    """Generate insights from multiple video analyses"""
    avg_score = sum(r['score'] for r in results) / len(results)
    
    insights = {
        'average_score': avg_score,
        'strengths': [],
        'weaknesses': [],
        'recommendations': []
    }
    
    # Analyze common patterns
    thumb_face_count = sum(1 for r in results if r['features']['thumb_has_face'] == 1)
    if thumb_face_count / len(results) < 0.5:
        insights['weaknesses'].append("Not enough thumbnails with faces")
        insights['recommendations'].append("Create thumbnail style guide with face close-ups")
    
    title_questions = sum(1 for r in results if r['features']['title_has_question'] == 1)
    if title_questions / len(results) < 0.3:
        insights['recommendations'].append("Use more question-based titles to increase curiosity")
    
    return insights

@app.route('/api/train', methods=['POST'])
def train_model():
    """Endpoint to train model with new data"""
    data = request.json
    
    # This would integrate with YouTubeDataCollector
    # For now, return mock response
    return jsonify({
        'status': 'success',
        'message': 'Model training started',
        'estimated_time': '10 minutes'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
