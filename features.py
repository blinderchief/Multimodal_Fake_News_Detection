"""
Feature Extraction Module
Why: Convert preprocessed data into numerical features for the model.
How: Use BERT for text embeddings, ViT for image features, and average video frames.
"""

import torch
from transformers import BertModel, ViTImageProcessor, ViTModel
from torchvision.transforms import ToTensor
import numpy as np

# Use CPU explicitly for Windows laptops without GPU
device = torch.device('cpu')

# Initialize models
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)

def get_text_features(text, max_length=512):
    """
    Extract text features using BERT.
    Why: BERT captures contextual meaning in text for better fake news detection.
    How: Tokenize text, pass through BERT, and get embeddings from the last hidden layer.
    """
    if not text:
        return torch.zeros(1, 768).to(device)

    # Tokenize the text
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # Get BERT embeddings
    with torch.no_grad():
        outputs = bert_model(**tokens)

    # Use mean pooling of the last hidden state
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Shape: [1, 768]
    return embeddings

def get_image_features(image):
    """
    Extract image features using Vision Transformer (ViT).
    Why: ViT is effective for image classification and can detect manipulations.
    How: Process image through ViT and get patch embeddings.
    """
    if image is None:
        return torch.zeros(1, 768).to(device)

    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Process image
    inputs = vit_processor(images=image, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get ViT features
    with torch.no_grad():
        outputs = vit_model(**inputs)

    # Use mean of patch embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Shape: [1, 768]
    return embeddings

def get_video_features(frames, max_frames=3):
    """
    Extract video features by averaging frame features.
    Why: Videos contain temporal information; averaging frames gives a summary.
    How: Extract features from each frame and average them.
    """
    if not frames:
        return torch.zeros(1, 768).to(device)

    # Limit number of frames for CPU efficiency
    frames = frames[:max_frames]
    frame_features = []

    for frame in frames:
        feat = get_image_features(frame)
        frame_features.append(feat)

    # Average the features
    if frame_features:
        video_features = torch.stack(frame_features).mean(dim=0)
    else:
        video_features = torch.zeros(1, 768).to(device)

    return video_features

def extract_multimodal_features(text, image=None, video_frames=None):
    """
    Extract features from all modalities.
    Why: Combine text, image, and video features for multimodal analysis.
    How: Call individual feature extractors and return a dictionary.
    """
    text_feat = get_text_features(text)
    img_feat = get_image_features(image)
    vid_feat = get_video_features(video_frames)

    return {
        'text': text_feat,
        'image': img_feat,
        'video': vid_feat
    }

if __name__ == "__main__":
    # Test feature extraction
    sample_text = "This is a test news article about fake news detection."
    text_features = get_text_features(sample_text)
    print("Text features shape:", text_features.shape)

    # Test with dummy image (224x224 random array)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image_features = get_image_features(dummy_image)
    print("Image features shape:", image_features.shape)