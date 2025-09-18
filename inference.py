"""
Inference Module with Gemini API
Why: Use Google's powerful multimodal model for fake news detection.
How: Send processed data to Gemini API and get predictions.
"""

import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

import google.generativeai as genai
from PIL import Image
import cv2
import numpy as np
from scrape import scrape_social_media

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
    # Initialize the model
    model = genai.GenerativeModel('gemini-2.0-flash')
else:
    model = None

def detect_fake_with_gemini(text, images=None, videos=None, social_url=None):
    """
    Detect fake news using Gemini API.
    Why: Gemini can analyze multimodal content and provide explanations.
    How: Craft a prompt with the content and get the model's response.
    """
    try:
        # Check if model is available
        if model is None:
            return "Error: Gemini API key not configured. Please set GEMINI_API_KEY in your .env file."

        # If social URL provided, scrape it first
        if social_url:
            scraped_text, scraped_images, scraped_videos = scrape_social_media(social_url)
            text = scraped_text or text
            images = (images or []) + scraped_images
            videos = (videos or []) + scraped_videos

        # Build the prompt
        prompt = f"""
        Analyze this news content for fake news detection. Provide:
        1. Verdict: REAL or FAKE
        2. Confidence score (0-100)
        3. Key reasons for your decision
        4. Any suspicious elements found

        News Text: {text}
        """

        # Prepare content for Gemini
        contents = [prompt]

        # Add images
        if images:
            for img in images[:3]:  # Limit for API
                if img is not None:
                    # Convert numpy array to PIL Image
                    if isinstance(img, np.ndarray):
                        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    else:
                        pil_img = img
                    contents.append(pil_img)

        # Add video frames
        if videos:
            for vid_frames in videos[:1]:  # Limit videos
                for frame in vid_frames[:3]:  # Limit frames
                    if isinstance(frame, np.ndarray):
                        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        contents.append(pil_frame)

        # Generate response
        response = model.generate_content(contents)

        return response.text

    except Exception as e:
        return f"Inference error: {e}"

def parse_gemini_response(response_text):
    """
    Parse the Gemini response to extract structured information.
    Why: Make the output easier to use in the UI.
    How: Extract verdict, confidence, and reasons.
    """
    try:
        lines = response_text.split('\n')
        verdict = "UNKNOWN"
        confidence = 0
        reasons = []

        for line in lines:
            line = line.strip()
            if "REAL" in line.upper():
                verdict = "REAL"
            elif "FAKE" in line.upper():
                verdict = "FAKE"
            elif "confidence" in line.lower() or "score" in line.lower():
                # Extract number
                import re
                numbers = re.findall(r'\d+', line)
                if numbers:
                    confidence = int(numbers[0])
            elif line.startswith("-") or line.startswith("*"):
                reasons.append(line[1:].strip())

        return {
            'verdict': verdict,
            'confidence': confidence,
            'reasons': reasons,
            'full_response': response_text
        }

    except Exception as e:
        return {
            'verdict': 'ERROR',
            'confidence': 0,
            'reasons': [str(e)],
            'full_response': response_text
        }

if __name__ == "__main__":
    # Test inference (requires GEMINI_API_KEY)
    test_text = "Breaking: Scientists discover aliens on Mars!"
    result = detect_fake_with_gemini(test_text)
    parsed = parse_gemini_response(result)
    print("Verdict:", parsed['verdict'])
    print("Confidence:", parsed['confidence'])
    print("Reasons:", parsed['reasons'])