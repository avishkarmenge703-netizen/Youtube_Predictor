import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import pytesseract
from typing import Dict, Tuple

class ThumbnailAnalyzer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def download_thumbnail(self, url: str) -> np.ndarray:
        """Download thumbnail image from URL"""
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    def analyze_thumbnail(self, image_url: str) -> Dict:
        """Extract features from thumbnail"""
        try:
            img = self.download_thumbnail(image_url)
            height, width = img.shape[:2]
            
            features = {
                'thumb_brightness': self.calculate_brightness(img),
                'thumb_contrast': self.calculate_contrast(img),
                'thumb_saturation': self.calculate_saturation(img),
                'thumb_has_face': self.detect_face(img),
                'thumb_face_count': self.count_faces(img),
                'thumb_has_text': self.detect_text(img),
                'text_area_ratio': self.text_area_ratio(img),
                'thumb_primary_color': self.get_dominant_color(img),
                'thumb_edges': self.edge_density(img),
                'thumb_entropy': self.calculate_entropy(img)
            }
            
            return features
        except Exception as e:
            print(f"Error analyzing thumbnail: {e}")
            return self.get_default_features()
    
    def calculate_brightness(self, img: np.ndarray) -> float:
        """Calculate average brightness"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return np.mean(hsv[:,:,2])
    
    def calculate_contrast(self, img: np.ndarray) -> float:
        """Calculate image contrast"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray.std()
    
    def calculate_saturation(self, img: np.ndarray) -> float:
        """Calculate average saturation"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return np.mean(hsv[:,:,1])
    
    def detect_face(self, img: np.ndarray) -> int:
        """Detect if face is present in thumbnail"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return 1 if len(faces) > 0 else 0
    
    def count_faces(self, img: np.ndarray) -> int:
        """Count number of faces"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces)
    
    def detect_text(self, img: np.ndarray) -> int:
        """Detect if text is present"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check for text-like contours
            text_contours = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Text typically has specific aspect ratios and sizes
                if 0.1 < aspect_ratio < 10 and w > 10 and h > 10:
                    text_contours += 1
            
            return 1 if text_contours > 2 else 0
        except:
            return 0
    
    def text_area_ratio(self, img: np.ndarray) -> float:
        """Calculate ratio of image area that might contain text"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        text_area = np.sum(edges > 0)
        total_area = img.shape[0] * img.shape[1]
        return text_area / total_area if total_area > 0 else 0
    
    def get_dominant_color(self, img: np.ndarray) -> int:
        """Get dominant color category (simplified)"""
        pixels = img.reshape(-1, 3)
        # Calculate average color
        avg_color = np.mean(pixels, axis=0)
        # Simple categorization
        if avg_color[2] > 150:  # Red channel
            return 1  # Warm
        elif avg_color[0] > 150:  # Blue channel
            return 2  # Cool
        else:
            return 0  # Neutral
    
    def edge_density(self, img: np.ndarray) -> float:
        """Calculate edge density"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return np.sum(edges > 0) / (img.shape[0] * img.shape[1])
    
    def calculate_entropy(self, img: np.ndarray) -> float:
        """Calculate image entropy"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def get_default_features(self) -> Dict:
        """Return default features if analysis fails"""
        return {key: 0 for key in [
            'thumb_brightness', 'thumb_contrast', 'thumb_saturation',
            'thumb_has_face', 'thumb_face_count', 'thumb_has_text',
            'text_area_ratio', 'thumb_primary_color', 'thumb_edges',
            'thumb_entropy'
        ]}
