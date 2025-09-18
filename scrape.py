"""
Real-Time Social Media Scraping Module
Why: Get fresh data from social media for real-time fake news detection.
How: Use Firecrawl SDK to scrape content from URLs or search queries.
"""

import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

import re
from firecrawl import Firecrawl
from preprocess import preprocess_text, preprocess_image, extract_frames

def scrape_social_media(url_or_query):
    """
    Scrape social media content.
    Why: Real-time data helps detect emerging fake news.
    How: Use Firecrawl to get markdown, then parse it.
    """
    try:
        # Check if API key is available
        api_key = os.getenv('FIRECRAWL_API_KEY')
        if not api_key:
            print("Warning: FIRECRAWL_API_KEY not found. Social media scraping disabled.")
            return "", [], []

        # Initialize Firecrawl with API key
        firecrawl = Firecrawl(api_key=api_key)

        if not url_or_query.startswith('http'):
            # Convert query to Twitter search URL
            url_or_query = f"https://x.com/search?q={url_or_query.replace(' ', '+')}"

        # Scrape the URL
        scrape_result = firecrawl.scrape(url_or_query, formats=['markdown'])

        # Get markdown content from the result - handle different response types
        try:
            if hasattr(scrape_result, 'markdown'):
                markdown = scrape_result.markdown
            elif isinstance(scrape_result, dict) and 'markdown' in scrape_result:
                markdown = scrape_result['markdown']
            else:
                print("No markdown content found in scrape result")
                return "", [], []
        except Exception as e:
            print(f"Error extracting markdown: {e}")
            return "", [], []

        # Parse the markdown
        parsed = parse_scraped_markdown(markdown)

        # Preprocess text
        proc_text = preprocess_text(parsed['text'])

        # Process images (limit to 2 for CPU)
        images = []
        for img_url in parsed['image_links'][:2]:
            img = preprocess_image(img_url)
            if img is not None:
                images.append(img)

        # Process videos (limit to 1 for CPU)
        videos = []
        for vid_url in parsed['video_links'][:1]:
            frames = extract_frames(vid_url, num_frames=3)
            if frames:
                videos.append(frames)

        return proc_text, images, videos

    except Exception as e:
        print(f"Scraping error: {e}")
        return "", [], []
    """
    Scrape social media content.
    Why: Real-time data helps detect emerging fake news.
    How: Use Firecrawl to get markdown, then parse it.
    """
    try:
        if not url_or_query.startswith('http'):
            # Convert query to Twitter search URL
            url_or_query = f"https://x.com/search?q={url_or_query.replace(' ', '+')}"

        # Scrape the URL
        scrape_result = firecrawl.scrape(url_or_query, formats=['markdown'])

        if 'markdown' not in scrape_result:
            print("No markdown content found")
            return "", [], []

        markdown = scrape_result['markdown']

        # Parse the markdown
        parsed = parse_scraped_markdown(markdown)

        # Preprocess text
        proc_text = preprocess_text(parsed['text'])

        # Process images (limit to 2 for CPU)
        images = []
        for img_url in parsed['image_links'][:2]:
            img = preprocess_image(img_url)
            if img is not None:
                images.append(img)

        # Process videos (limit to 1 for CPU)
        videos = []
        for vid_url in parsed['video_links'][:1]:
            frames = extract_frames(vid_url, num_frames=3)
            if frames:
                videos.append(frames)

        return proc_text, images, videos

    except Exception as e:
        print(f"Scraping error: {e}")
        return "", [], []

def parse_scraped_markdown(markdown_text):
    """
    Parse markdown content from scraped data.
    Why: Extract clean text and media links.
    How: Use regex patterns to find different elements.
    """
    # Extract text (remove image/video links)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove other links
    text = re.sub(r'#+\s*', '', text)  # Remove headers

    # Extract image links
    image_links = re.findall(r'!\[.*?\]\((https?://.*?(\.jpg|\.png|\.gif|\.jpeg))\)', markdown_text)

    # Extract video links
    video_links = re.findall(r'!\[.*?\]\((https?://.*?(\.mp4|\.webm|\.avi|\.mov))\)', markdown_text)

    return {
        'text': text.strip(),
        'image_links': [link[0] for link in image_links],
        'video_links': [link[0] for link in video_links]
    }

if __name__ == "__main__":
    # Test scraping (requires API key)
    # Note: This will only work if you have set FIRECRAWL_API_KEY in .env
    test_query = "fake news"
    text, images, videos = scrape_social_media(test_query)
    print("Scraped text:", text[:200] + "..." if text else "No text")
    print("Images found:", len(images))
    print("Videos found:", len(videos))