"""
Preprocessing Pipeline Module
Why: Raw data needs cleaning and standardization before feature extraction.
How: Clean text, resize images, extract video frames, and parse scraped markdown.
"""

import cv2
from transformers import AutoTokenizer
import nltk
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import re

# Download NLTK stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

# Initialize BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    """
    Preprocess text data.
    Why: Remove noise and standardize text for better model performance.
    How: Convert to lowercase, remove stopwords, tokenize.
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Split into words
    words = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    # Join back
    return ' '.join(words)

def preprocess_image(image_path_or_url):
    """
    Preprocess image data.
    Why: Standardize image size for consistent feature extraction.
    How: Resize to 224x224 pixels, handle both local files and URLs.
    """
    try:
        if image_path_or_url.startswith('http'):
            # Download image from URL
            response = requests.get(image_path_or_url, timeout=5)
            img = Image.open(BytesIO(response.content))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            # Load local image
            img = cv2.imread(image_path_or_url)

        if img is None:
            return None

        # Resize to standard size for ViT
        img = cv2.resize(img, (224, 224))
        return img
    except Exception as e:
        print(f"Image processing error: {e}")
        return None

def extract_frames(video_path_or_url, num_frames=5):
    """
    Extract frames from video.
    Why: Videos are sequences; we extract key frames for analysis.
    How: Sample frames evenly across the video duration.
    """
    try:
        if video_path_or_url.startswith('http'):
            # Download video from URL
            response = requests.get(video_path_or_url, timeout=10)
            with open('temp_video.mp4', 'wb') as f:
                f.write(response.content)
            video_path = 'temp_video.mp4'
        else:
            video_path = video_path_or_url

        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return []

        # Calculate step size for even sampling
        step = max(1, total_frames // num_frames)

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
                if len(frames) >= num_frames:
                    break

        cap.release()
        return frames
    except Exception as e:
        print(f"Video processing error: {e}")
        return []

def parse_scraped_markdown(markdown_text):
    """
    Parse markdown from scraped social media content.
    Why: Extract clean text and media links from raw markdown.
    How: Use regex to find text and image/video URLs.
    """
    # Extract text (remove markdown formatting)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'#+\s*', '', text)

    # Extract image links
    image_links = re.findall(r'!\[.*?\]\((https?://.*?(\.jpg|\.png|\.gif|\.jpeg))\)', markdown_text)

    # Extract video links
    video_links = re.findall(r'!\[.*?\]\((https?://.*?(\.mp4|\.webm|\.avi))\)', markdown_text)

    return {
        'text': text.strip(),
        'image_links': [link[0] for link in image_links],
        'video_links': [link[0] for link in video_links]
    }

if __name__ == "__main__":
    # Test preprocessing functions
    sample_text = "Breaking: Alien invasion detected! This is fake news."
    processed_text = preprocess_text(sample_text)
    print("Original text:", sample_text)
    print("Processed text:", processed_text)

    # Test image preprocessing (if you have an image URL)
    # img = preprocess_image('https://example.com/image.jpg')
    # print("Image shape:", img.shape if img is not None else "None")