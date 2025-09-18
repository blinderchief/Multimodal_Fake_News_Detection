# Multimodal Fake News Detection System

A comprehensive AI-powered system for detecting fake news across text, images, and videos using multimodal analysis and real-time social media scraping.

## 🚀 Features

- **Multimodal Analysis**: Detect fake news in text, images, and videos
- **Real-Time Scraping**: Scrape social media content using Firecrawl
- **Cloud Inference**: Use Google Gemini 2.0 API for powerful detection
- **Explainable AI**: Visual explanations of model decisions
- **Professional UI**: Goldman Sachs-inspired web interface
- **Graph Neural Networks**: Cross-modal consistency checking
- **CPU Optimized**: Works on laptops without GPU

## 🏗️ Architecture

```
Input (Text + Image/Video + Social Media URL) → Real-Time Scraping (Firecrawl)
→ Preprocessing → Feature Extraction (BERT + ViT) → Graph Fusion (PyG GNN)
→ Gemini API Inference → Explainable AI → Professional UI Output
```

## 📋 Prerequisites

- **Python 3.12+**
- **Windows/Linux/Mac**
- **8GB+ RAM** (16GB recommended)
- **API Keys**:
  - Google Gemini API ([ai.google.dev](https://ai.google.dev))
  - Firecrawl API ([firecrawl.dev](https://firecrawl.dev))
  - Optional: Weights & Biases ([wandb.ai](https://wandb.ai))

## 🛠️ Installation

### Using UV (Recommended)

1. **Install UV**:
   ```bash
   pip install uv
   ```

2. **Clone/Setup Project**:
   ```bash
   cd multimodal_fake_news
   uv sync
   ```

3. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Alternative (pip)

```bash
pip install torch torchvision torchaudio transformers opencv-python torch-geometric google-generativeai streamlit captum pandas numpy wandb firecrawl-py requests python-dotenv datasets nltk
```

## 🚀 Usage

### Web Application

```bash
uv run streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Training the Model

```bash
uv run python train_eval.py
```

### API Usage

```python
from inference import detect_fake_with_gemini

result = detect_fake_with_gemini(
    "Breaking news: Aliens landed!",
    social_url="https://x.com/search?q=aliens"
)
print(result)
```

## 📁 Project Structure

```
multimodal_fake_news/
├── app.py                 # Streamlit web application
├── data.py               # Data loading and exploration
├── preprocess.py         # Data preprocessing pipeline
├── features.py           # Feature extraction (BERT, ViT)
├── graph_fusion.py       # Graph neural network fusion
├── scrape.py             # Social media scraping
├── inference.py          # Gemini API integration
├── xai.py                # Explainable AI
├── train_eval.py         # Model training and evaluation
├── Dockerfile            # Container deployment
├── pyproject.toml        # Project dependencies
├── .env.example          # Environment variables template
└── README.md            # This file
```

## 🎯 Key Components

### 1. Data Preprocessing (`preprocess.py`)
- **Text Cleaning**: Remove stopwords, normalize text
- **Image Processing**: Resize to 224x224, handle URLs
- **Video Processing**: Extract key frames (CPU-optimized)

### 2. Feature Extraction (`features.py`)
- **Text Features**: BERT embeddings for semantic understanding
- **Visual Features**: ViT for image/video analysis
- **Multimodal Fusion**: Combine features for comprehensive analysis

### 3. Graph Neural Networks (`graph_fusion.py`)
- **Cross-Modal Consistency**: Detect inconsistencies between modalities
- **GNN Architecture**: PyTorch Geometric implementation
- **Fake Detection**: Graph-based prediction

### 4. Social Media Scraping (`scrape.py`)
- **Firecrawl Integration**: Real-time content scraping
- **Multi-Platform**: Twitter/X, Reddit, news sites
- **Content Parsing**: Extract text, images, videos

### 5. AI Inference (`inference.py`)
- **Gemini API**: Cloud-based multimodal analysis
- **Prompt Engineering**: Optimized for fake news detection
- **Response Parsing**: Structured output extraction

### 6. Explainable AI (`xai.py`)
- **Integrated Gradients**: Feature attribution
- **Visual Explanations**: Heatmaps for suspicious regions
- **Text Highlighting**: Key suspicious words

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```bash
GEMINI_API_KEY=your_gemini_key
FIRECRAWL_API_KEY=your_firecrawl_key
WANDB_API_KEY=your_wandb_key  # Optional
```

### Model Configuration

- **CPU Optimization**: All models run on CPU
- **Batch Size**: Limited to 1 for memory efficiency
- **Frame Limit**: 3-5 frames per video
- **Image Size**: 224x224 pixels

## 📊 Performance

### Expected Metrics
- **Accuracy**: 85%+ on test datasets
- **F1-Score**: 0.82+ (balanced precision/recall)
- **Inference Time**: 2-5 seconds per analysis
- **Scraping Time**: 1-3 seconds per URL

### CPU Performance Notes
- Feature extraction: ~1-2 minutes per sample
- Training: ~5-10 minutes per epoch
- Inference: Near real-time with Gemini API

## 🚀 Deployment

### Docker Deployment

```bash
# Build image
docker build -t fake-news-detector .

# Run container
docker run -p 8080:8080 fake-news-detector
```

### Local Deployment

```bash
# Install dependencies
uv sync

# Run application
uv run streamlit run app.py --server.port=8080
```

## 🧪 Testing

### Unit Tests

```bash
uv run python -m pytest tests/
```

### Manual Testing

1. **Text Analysis**: Input news article text
2. **Image Analysis**: Upload suspicious image
3. **Video Analysis**: Upload video file
4. **Social Media**: Enter URL or search query
5. **Combined**: Test multimodal inputs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please respect API terms of service and content policies.

## 🙏 Acknowledgments

- **Google Gemini**: For multimodal AI capabilities
- **Firecrawl**: For social media scraping
- **Hugging Face**: For pre-trained models
- **PyTorch**: For deep learning framework

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Open an issue on GitHub
3. Review the PRD documentation

---

**Built with ❤️ for combating misinformation**