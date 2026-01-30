import os
import pandas as pd
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import time
from typing import List, Dict
import json

class YouTubeDataCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
    
    def get_channel_videos(self, channel_id: str, max_results: int = 50):
        """Get all videos from a channel"""
        videos = []
        request = self.youtube.search().list(
            part="snippet",
            channelId=channel_id,
            maxResults=50,
            order="date",
            type="video"
        )
        
        while request and len(videos) < max_results:
            response = request.execute()
            
            for item in response['items']:
                video_id = item['id']['videoId']
                video_data = self.get_video_details(video_id)
                if video_data:
                    videos.append(video_data)
            
            # Get next page if available
            request = self.youtube.search().list_next(request, response)
            time.sleep(0.1)  # Respect API limits
        
        return pd.DataFrame(videos)
    
    def get_video_details(self, video_id: str) -> Dict:
        """Get comprehensive video data"""
        try:
            # Get video statistics and snippet
            response = self.youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=video_id
            ).execute()
            
            if not response['items']:
                return None
            
            item = response['items'][0]
            snippet = item['snippet']
            stats = item['statistics']
            
            # Calculate 48-hour performance (approximation)
            published_at = datetime.fromisoformat(
                snippet['publishedAt'].replace('Z', '+00:00')
            )
            current_time = datetime.utcnow()
            hours_since_publish = (current_time - published_at).total_seconds() / 3600
            
            # Estimate first 48-hour views (simple extrapolation)
            if 'viewCount' in stats:
                view_count = int(stats['viewCount'])
                view_rate = view_count / max(hours_since_publish, 1)
                estimated_48h_views = min(view_rate * 48, view_count)
            else:
                estimated_48h_views = 0
            
            # Calculate CTR (if impressions available)
            # Note: Impressions not available in public API
            # Using engagement ratio as proxy
            if 'viewCount' in stats and 'likeCount' in stats:
                ctr_proxy = (int(stats.get('likeCount', 0)) + 
                           int(stats.get('commentCount', 0))) / max(int(stats['viewCount']), 1)
            else:
                ctr_proxy = 0
            
            video_data = {
                'video_id': video_id,
                'title': snippet['title'],
                'description': snippet.get('description', ''),
                'tags': ','.join(snippet.get('tags', [])),
                'published_at': snippet['publishedAt'],
                'category_id': snippet.get('categoryId'),
                'duration': item['contentDetails'].get('duration'),
                'view_count': int(stats.get('viewCount', 0)),
                'like_count': int(stats.get('likeCount', 0)),
                'comment_count': int(stats.get('commentCount', 0)),
                'estimated_48h_views': estimated_48h_views,
                'ctr_proxy': ctr_proxy,
                'thumbnail_url': snippet['thumbnails']['high']['url']
            }
            
            return video_data
            
        except Exception as e:
            print(f"Error fetching video {video_id}: {e}")
            return None
    
    def search_videos_by_keyword(self, keyword: str, max_results: int = 100):
        """Search for videos by keyword in specific niche"""
        videos = []
        request = self.youtube.search().list(
            part="snippet",
            q=keyword,
            maxResults=50,
            order="viewCount",
            type="video",
            videoDuration="medium"  # medium length videos (4-20 minutes)
        )
        
        while request and len(videos) < max_results:
            response = request.execute()
            
            for item in response['items']:
                video_id = item['id']['videoId']
                video_data = self.get_video_details(video_id)
                if video_data:
                    videos.append(video_data)
            
            request = self.youtube.search().list_next(request, response)
            time.sleep(0.1)
        
        return pd.DataFrame(videos)
    
    def save_training_data(self, df: pd.DataFrame, filename: str = "training_data.csv"):
        """Save collected data to CSV"""
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} videos to {filename}")
