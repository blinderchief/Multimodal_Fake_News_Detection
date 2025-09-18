"""
Cross-Modal Graph Fusion Module
Why: Detect inconsistencies between modalities using graph neural networks.
How: Build a graph with features as nodes, similarities as edges, and use GNN for fusion.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class GraphFakeDetector(torch.nn.Module):
    """
    Graph Neural Network for fake news detection.
    Why: GNNs can model relationships between different modalities.
    How: Use GCN layers to propagate information across the graph.
    """
    def __init__(self, input_dim=768, hidden_dim=128, output_dim=2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # First GCN layer
        x = torch.relu(self.conv1(x, edge_index))
        # Second GCN layer
        x = self.conv2(x, edge_index)
        # Apply softmax for classification
        return torch.softmax(x, dim=-1)

def build_graph(text_feat, img_feat, vid_feat):
    """
    Build a graph from multimodal features.
    Why: Represent cross-modal relationships as a graph.
    How: Use features as nodes, cosine similarity as edges.
    """
    # Concatenate all features
    feats = torch.cat([text_feat, img_feat, vid_feat], dim=0).cpu().numpy()

    # Calculate cosine similarity between all pairs
    sim_matrix = cosine_similarity(feats)

    # Create edges for high similarity (> 0.5)
    edge_index = []
    for i in range(len(feats)):
        for j in range(len(feats)):
            if i != j and sim_matrix[i, j] > 0.5:
                edge_index.append([i, j])

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    else:
        # If no edges, create self-loops
        edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)

    # Create PyTorch Geometric Data object
    data = Data(x=torch.tensor(feats, dtype=torch.float), edge_index=edge_index)
    return data

def detect_fake_with_graph(text_feat, img_feat, vid_feat, model=None):
    """
    Detect fake news using the graph-based approach.
    Why: Combine modalities to find inconsistencies.
    How: Build graph, pass through GNN, get prediction.
    """
    if model is None:
        model = GraphFakeDetector().to('cpu')

    # Build graph
    graph_data = build_graph(text_feat, img_feat, vid_feat)

    # Make prediction
    with torch.no_grad():
        output = model(graph_data.x, graph_data.edge_index)

    # Get prediction for the text node (assuming text is node 0)
    prediction = output[0].argmax().item()
    confidence = output[0].max().item()

    return prediction, confidence  # 0: real, 1: fake

if __name__ == "__main__":
    # Test graph fusion
    # Create dummy features
    text_feat = torch.randn(1, 768)
    img_feat = torch.randn(1, 768)
    vid_feat = torch.randn(1, 768)

    # Build graph
    graph = build_graph(text_feat, img_feat, vid_feat)
    print("Graph nodes:", graph.x.shape)
    print("Graph edges:", graph.edge_index.shape)

    # Test model
    model = GraphFakeDetector()
    pred, conf = detect_fake_with_graph(text_feat, img_feat, vid_feat, model)
    print(f"Prediction: {'Fake' if pred else 'Real'}, Confidence: {conf:.2f}")