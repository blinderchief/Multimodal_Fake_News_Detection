"""
Training and Evaluation Module
Why: Train the GNN model and evaluate its performance on test data.
How: Use PyTorch for training loop and scikit-learn for metrics.
"""

import torch
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import wandb
from graph_fusion import GraphFakeDetector, build_graph
from data import load_fake_news_dataset
from preprocess import preprocess_text, preprocess_image
from features import get_text_features, get_image_features, get_video_features

# Set device
device = torch.device('cpu')

def prepare_training_data(dataset_df, max_samples=100):
    """
    Prepare data for training.
    Why: Convert raw data to graph format for GNN training.
    How: Extract features and build graphs for each sample.
    """
    graphs = []
    labels = []

    for idx, row in dataset_df.iterrows():
        if len(graphs) >= max_samples:
            break

        try:
            # Preprocess text
            text = preprocess_text(row.get('text', ''))

            # Preprocess image (if available)
            image = None
            if 'image_url' in row and row['image_url']:
                image = preprocess_image(row['image_url'])

            # For simplicity, assume no video in dataset
            video_frames = None

            # Extract features
            text_feat = get_text_features(text)
            img_feat = get_image_features(image)
            vid_feat = get_video_features(video_frames)

            # Build graph
            graph = build_graph(text_feat, img_feat, vid_feat)
            graphs.append(graph)

            # Get label
            label = row.get('label', 0)  # 0: real, 1: fake
            labels.append(label)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    return graphs, labels

def train_model(graphs, labels, epochs=5, lr=0.001):
    """
    Train the GNN model.
    Why: Learn to detect fake news from multimodal features.
    How: Use Adam optimizer and cross-entropy loss.
    """
    model = GraphFakeDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Convert labels to tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)

    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for i, graph in enumerate(graphs):
            optimizer.zero_grad()

            # Forward pass
            output = model(graph.x.to(device), graph.edge_index.to(device))

            # For simplicity, use prediction from first node
            pred = output[0].unsqueeze(0)
            target = labels_tensor[i].unsqueeze(0)

            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(graphs)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Log to wandb if available
        if wandb.run:
            wandb.log({'epoch': epoch, 'loss': avg_loss})

    return model

def evaluate_model(model, test_graphs, test_labels):
    """
    Evaluate the trained model.
    Why: Measure performance on unseen data.
    How: Calculate accuracy, F1-score, and other metrics.
    """
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for graph, label in zip(test_graphs, test_labels):
            output = model(graph.x.to(device), graph.edge_index.to(device))
            pred = output[0].argmax().item()
            predictions.append(pred)
            true_labels.append(label)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')

    print("Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=['Real', 'Fake']))

    # Log to wandb
    if wandb.run:
        wandb.log({
            'test_accuracy': accuracy,
            'test_f1': f1
        })

    return accuracy, f1

def main():
    """
    Main training and evaluation pipeline.
    """
    # Initialize wandb (optional)
    try:
        wandb.init(project='multimodal-fake-news', name='gnn-training')
    except:
        print("Wandb not configured, skipping logging")

    # Load dataset
    print("Loading dataset...")
    df = load_fake_news_dataset()
    if df is None or df.empty:
        print("No dataset available. Please check your data source.")
        return

    # Split data (simple split for demo)
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]

    # Prepare training data
    print("Preparing training data...")
    train_graphs, train_labels = prepare_training_data(train_df, max_samples=50)  # Small for CPU
    test_graphs, test_labels = prepare_training_data(test_df, max_samples=20)

    if not train_graphs:
        print("No training data prepared.")
        return

    # Train model
    print("Training model...")
    model = train_model(train_graphs, train_labels, epochs=3)  # Reduced for CPU

    # Evaluate
    print("Evaluating model...")
    accuracy, f1 = evaluate_model(model, test_graphs, test_labels)

    # Save model
    torch.save(model.state_dict(), 'fake_news_detector.pth')
    print("Model saved as 'fake_news_detector.pth'")

    if wandb.run:
        wandb.finish()

if __name__ == "__main__":
    main()