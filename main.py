"""
Main Entry Point for Multimodal Fake News Detection System
Why: Provide a simple way to run the entire system.
How: Import and orchestrate all modules.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if all required API keys are configured."""
    required_keys = ['GEMINI_API_KEY', 'FIRECRAWL_API_KEY']
    missing_keys = []

    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)

    if missing_keys:
        print("⚠️  Missing API keys. Please configure the following in your .env file:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nCopy .env.example to .env and fill in your keys.")
        return False

    print("✅ Environment configured successfully!")
    return True

def main():
    """Main function to run the system."""
    print("📰 Multimodal Fake News Detection System")
    print("=" * 50)

    if not check_environment():
        return

    print("\n🚀 Starting system...")
    print("Available options:")
    print("1. Run Web Application (uv run streamlit run app.py)")
    print("2. Train Model (uv run python train_eval.py)")
    print("3. Test Inference (uv run python inference.py)")
    print("4. Explore Data (uv run python data.py)")

    print("\n💡 Tip: Use 'uv run' for better dependency management")
    print("   Example: uv run streamlit run app.py")

if __name__ == "__main__":
    main()
