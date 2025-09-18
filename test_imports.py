#!/usr/bin/env python3
"""
Test script to verify imports work without API keys
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from inference import detect_fake_with_gemini, parse_gemini_response
    print("✅ Inference module imported successfully")

    # Test the function without API key
    result = detect_fake_with_gemini("test text")
    print(f"Function result: {result}")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Runtime error: {e}")

try:
    from scrape import scrape_social_media
    print("✅ Scrape module imported successfully")

    # Test the function without API key
    result = scrape_social_media("test query")
    print(f"Function result: {result}")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Runtime error: {e}")

print("\n🎉 Test completed!")