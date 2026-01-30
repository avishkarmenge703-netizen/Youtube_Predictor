import re
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, List, Tuple
import numpy as np

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

class TextAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.curiosity_words = [
            'secret', 'revealed', 'hidden', 'truth', 'exposed',
            'shocking', 'unexpected', 'mystery', 'why', 'how',
            'what if', 'never seen', 'breaking', 'urgent'
        ]
        self.power_words = [
            'free', 'instantly', 'guaranteed', 'proven', 'ultimate',
            'essential', 'powerful', 'massive', 'quick', 'easy',
            'simple', 'fast', 'huge', 'exclusive', 'limited'
        ]
        self.cta_phrases = [
            'click here', 'subscribe', 'like', 'comment',
            'watch now', 'learn more', 'check out', 'don\'t miss',
            'join now', 'get started'
        ]
    
    def analyze_title(self, title: str) -> Dict:
        """Extract features from title"""
        features = {
            'title_length': len(title),
            'title_word_count': len(title.split()),
            'title_has_question': 1 if '?' in title else 0,
            'title_has_number': 1 if bool(re.search(r'\d', title)) else 0,
            'title_has_brackets': 1 if '[' in title or '(' in title else 0,
            'title_curiosity_score': self.calculate_curiosity_score(title),
            'title_power_score': self.calculate_power_word_score(title),
            'title_sentiment': self.analyze_sentiment(title),
            'title_capital_ratio': self.capitalization_ratio(title),
            'title_emoji_count': self.count_emojis(title)
        }
        return features
    
    def analyze_description(self, description: str) -> Dict:
        """Extract features from description"""
        features = {
            'desc_length': len(description),
            'desc_word_count': len(description.split()),
            'desc_has_timestamps': 1 if bool(re.search(r'\d+:\d+', description)) else 0,
            'desc_has_links': 1 if 'http://' in description or 'https://' in description else 0,
            'desc_link_count': len(re.findall(r'https?://\S+', description)),
            'desc_cta_count': self.count_cta_phrases(description),
            'desc_has_hashtags': 1 if '#' in description else 0,
            'desc_has_bullets': 1 if '•' in description or '-' in description else 0,
            'desc_sentiment': self.analyze_sentiment(description),
            'desc_readability_score': self.flesch_reading_ease(description)
        }
        return features
    
    def analyze_tags(self, tags_str: str) -> Dict:
        """Extract features from tags"""
        if not tags_str:
            return {'tag_count': 0, 'tag_avg_length': 0, 'tag_relevance': 0}
        
        tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
        
        features = {
            'tag_count': len(tags),
            'tag_avg_length': np.mean([len(tag) for tag in tags]) if tags else 0,
            'tag_has_brand': 1 if any(self.is_brand_name(tag) for tag in tags) else 0,
            'tag_diversity': len(set(tags)) / len(tags) if tags else 0
        }
        return features
    
    def calculate_curiosity_score(self, text: str) -> float:
        """Calculate curiosity gap score"""
        score = 0
        words = text.lower().split()
        for word in self.curiosity_words:
            if word in text.lower():
                score += 1
        return score / max(len(words), 1)
    
    def calculate_power_word_score(self, text: str) -> float:
        """Calculate power word score"""
        score = 0
        words = text.lower().split()
        for word in self.power_words:
            if word in text.lower():
                score += 1
        return score / max(len(words), 1)
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment polarity (-1 to 1)"""
        scores = self.sia.polarity_scores(text)
        return scores['compound']
    
    def count_cta_phrases(self, text: str) -> int:
        """Count Call-to-Action phrases"""
        count = 0
        text_lower = text.lower()
        for phrase in self.cta_phrases:
            if phrase in text_lower:
                count += 1
        return count
    
    def capitalization_ratio(self, text: str) -> float:
        """Calculate ratio of capitalized words"""
        words = text.split()
        if not words:
            return 0
        capitalized = sum(1 for word in words if word and word[0].isupper())
        return capitalized / len(words)
    
    def count_emojis(self, text: str) -> int:
        """Count number of emojis"""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE
        )
        return len(emoji_pattern.findall(text))
    
    def flesch_reading_ease(self, text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        if not sentences or not words:
            return 0
        
        total_sentences = len(sentences)
        total_words = len(words)
        total_syllables = sum(self.count_syllables(word) for word in words)
        
        if total_words == 0 or total_sentences == 0:
            return 0
        
        score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
        return max(0, min(100, score))
    
    def count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximate)"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count
    
    def is_brand_name(self, tag: str) -> bool:
        """Check if tag contains brand name patterns"""
        brand_indicators = ['official', 'channel', 'tv', 'hd', '4k', 'studio']
        return any(indicator in tag.lower() for indicator in brand_indicators)
