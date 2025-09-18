"""
Streamlit UI Application
Why: Provide a user-friendly interface for the fake news detection system.
How: Use Streamlit with custom CSS styling.
"""

import streamlit as st
from preprocess import preprocess_text, preprocess_image, extract_frames
import os
from PIL import Image
import cv2
import numpy as np

# Try to import inference modules, handle missing API keys gracefully
try:
    from inference import detect_fake_with_gemini, parse_gemini_response
    INFERENCE_AVAILABLE = True
except ImportError as e:
    st.warning("⚠️ Inference module not available. Please configure API keys in .env file.")
    INFERENCE_AVAILABLE = False
    detect_fake_with_gemini = None
    parse_gemini_response = None

# Page configuration
st.set_page_config(
    page_title="Multimodal Fake News Detector",
    page_icon="📰",
    layout="wide"
)

st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #FFFFFF;
        color: #333333;
    }

    /* Sidebar styling - Multiple selectors for compatibility */
    [data-testid="stSidebar"],
    .css-1d391kg,  /* Streamlit sidebar container */
    .css-1lcbmhc,  /* Alternative sidebar selector */
    .sidebar,
    .stSidebar {
        background-color: #001F3F !important;
    }

    /* Sidebar content area */
    [data-testid="stSidebar"] > div,
    .css-1d391kg > div,
    .css-1lcbmhc > div,
    .sidebar .sidebar-content,
    .stSidebar > div {
        background-color: #001F3F !important;
        color: #FFFFFF !important;
    }

    /* Sidebar text elements - Comprehensive selectors */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stText,
    .css-1d391kg h1,
    .css-1d391kg h2,
    .css-1d391kg h3,
    .css-1d391kg h4,
    .css-1d391kg p,
    .css-1d391kg div,
    .css-1d391kg span,
    .css-1d391kg .stMarkdown,
    .css-1d391kg .stText,
    .css-1lcbmhc h1,
    .css-1lcbmhc h2,
    .css-1lcbmhc h3,
    .css-1lcbmhc h4,
    .css-1lcbmhc p,
    .css-1lcbmhc div,
    .css-1lcbmhc span,
    .css-1lcbmhc .stMarkdown,
    .css-1lcbmhc .stText,
    .sidebar .sidebar-content h1,
    .sidebar .sidebar-content h2,
    .sidebar .sidebar-content h3,
    .sidebar .sidebar-content h4,
    .sidebar .sidebar-content p,
    .sidebar .sidebar-content div,
    .sidebar .sidebar-content span,
    .sidebar .sidebar-content .stMarkdown,
    .sidebar .sidebar-content .stText {
        color: #FFFFFF !important;
        background-color: transparent !important;
    }

    /* Sidebar markdown content specifically */
    [data-testid="stSidebar"] .stMarkdown div,
    .css-1d391kg .stMarkdown div,
    .css-1lcbmhc .stMarkdown div,
    .sidebar .stMarkdown div {
        color: #FFFFFF !important;
        background-color: transparent !important;
    }

    /* Sidebar list items and bullet points */
    [data-testid="stSidebar"] li,
    .css-1d391kg li,
    .css-1lcbmhc li,
    .sidebar li {
        color: #FFFFFF !important;
    }

    /* Sidebar strong/bold text */
    [data-testid="stSidebar"] strong,
    [data-testid="stSidebar"] b,
    .css-1d391kg strong,
    .css-1d391kg b,
    .css-1lcbmhc strong,
    .css-1lcbmhc b,
    .sidebar strong,
    .sidebar b {
        color: #FFFFFF !important;
        font-weight: bold !important;
    }

    /* Button styling */
    .stButton > button {
        background-color: #FFD700;
        color: #000000 !important;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        border: 2px solid #FFD700;
    }

    .stButton > button:hover {
        background-color: #FFC107;
        border-color: #FFC107;
    }

    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #001F3F !important;
        font-weight: 600;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        background-color: #F0F8FF;
        color: #001F3F !important;
        border-radius: 5px;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E6F3FF;
    }

    /* Active tab styling */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #001F3F !important;
        color: #FFFFFF !important;
    }

    /* Input field styling */
    .stTextInput input,
    .stTextArea textarea,
    .stFileUploader div {
        color: #333333 !important;
        background-color: #FFFFFF !important;
        border: 1px solid #CCCCCC !important;
    }

    /* Verdict styling */
    .verdict-real {
        color: #28a745 !important;
        font-size: 24px;
        font-weight: bold;
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        border: 2px solid #28a745;
    }

    .verdict-fake {
        color: #dc3545 !important;
        font-size: 24px;
        font-weight: bold;
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        border: 2px solid #dc3545;
    }

    /* Success and error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        color: #333333 !important;
    }

    /* Metric styling */
    .stMetric {
        background-color: #F8F9FA;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #E9ECEF;
    }

    /* Expander styling */
    .stExpander {
        background-color: #FFFFFF;
        border: 1px solid #E9ECEF;
        border-radius: 5px;
    }

    /* Footer styling */
    .stMarkdown p:last-child {
        color: #666666 !important;
        font-size: 14px;
    }

    /* General text color override */
    .stMarkdown, .stText, .stCaption {
        color: #333333 !important;
    }

    /* Link styling */
    a {
        color: #007BFF !important;
    }

    /* Spinner styling */
    .stSpinner {
        color: #001F3F !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📰 Multimodal Fake News Detector")
    st.markdown("AI-Powered Detection System")
    st.markdown("---")
    st.markdown("**Features:**")
    st.markdown("- Text Analysis")
    st.markdown("- Image Verification")
    st.markdown("- Video Analysis")
    st.markdown("- Social Media Scraping")
    st.markdown("- Explainable AI")

# Main content
st.title("📰 Multimodal Fake News Detection System")

# Tabs for different input methods
tab1, tab2 = st.tabs(["📝 Manual Input", "🌐 Social Media Scraping"])

# Manual Input Tab
with tab1:
    st.header("Manual Input Analysis")

    # Text input
    text_input = st.text_area(
        "Enter news text:",
        height=100,
        placeholder="Paste the news article text here..."
    )

    # Image upload
    uploaded_image = st.file_uploader(
        "Upload image (optional)",
        type=['jpg', 'png', 'jpeg', 'gif']
    )

    # Video upload
    uploaded_video = st.file_uploader(
        "Upload video (optional)",
        type=['mp4', 'avi', 'mov']
    )

# Social Media Tab
with tab2:
    st.header("Social Media Analysis")

    social_url = st.text_input(
        "Enter social media URL or search query:",
        placeholder="e.g., https://x.com/search?q=fake+news or 'breaking news aliens'"
    )

    st.markdown("*Tip: For Twitter/X, you can use search queries like 'fake news election'*")

# Analysis button
if st.button("🔍 Analyze Content", type="primary"):
    with st.spinner("Analyzing content... This may take a few moments."):

        # Process inputs
        processed_text = ""
        processed_image = None
        processed_video = None

        # Handle manual inputs
        if text_input:
            processed_text = preprocess_text(text_input)

        if uploaded_image:
            # Convert uploaded file to numpy array
            image = Image.open(uploaded_image)
            processed_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            processed_image = cv2.resize(processed_image, (224, 224))

        if uploaded_video:
            # Save uploaded video temporarily
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_video.read())
            processed_video = extract_frames("temp_video.mp4", num_frames=3)

        # Perform inference
        if not INFERENCE_AVAILABLE or detect_fake_with_gemini is None:
            st.error("Analysis not available. Please configure your API keys in the .env file.")
            st.info("Required: GEMINI_API_KEY and FIRECRAWL_API_KEY")
        else:
            try:
                result = detect_fake_with_gemini(
                    processed_text,
                    [processed_image] if processed_image is not None else None,
                    [processed_video] if processed_video else None,
                    social_url if social_url else None
                )

                if parse_gemini_response:
                    parsed_result = parse_gemini_response(result)
                else:
                    parsed_result = {'verdict': 'ERROR', 'confidence': 0, 'reasons': ['Parse function not available'], 'full_response': result}

                # Display results
                st.success("Analysis Complete!")

                # Verdict
                col1, col2 = st.columns(2)
                with col1:
                    if parsed_result['verdict'] == 'FAKE':
                        st.markdown(f"<div class='verdict-fake'>🚨 VERDICT: FAKE NEWS</div>", unsafe_allow_html=True)
                    elif parsed_result['verdict'] == 'REAL':
                        st.markdown(f"<div class='verdict-real'>✅ VERDICT: REAL NEWS</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("🤔 VERDICT: UNCERTAIN")

                with col2:
                    st.metric("Confidence Score", f"{parsed_result['confidence']}%")

                # Reasons
                if parsed_result['reasons']:
                    st.subheader("🔍 Key Findings:")
                    for reason in parsed_result['reasons']:
                        st.write(f"• {reason}")

                # Full response
                with st.expander("📄 Full Analysis"):
                    st.write(parsed_result['full_response'])

                # Generate explanation if image provided
                if processed_image is not None:
                    st.subheader("🎨 Visual Explanation")
                    # This would generate and display the explanation heatmap
                    st.info("Visual explanation feature available in full implementation")

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.info("Make sure your API keys are configured in the .env file")

# Footer
st.markdown("---")
st.markdown("*Built with PyTorch, Transformers, and Google Gemini API*")
st.markdown("*For educational and research purposes only*")

if __name__ == "__main__":
    # This will run when executed directly
    st.write("Run `streamlit run app.py` to start the web application")