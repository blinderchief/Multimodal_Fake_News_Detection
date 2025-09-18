"""
Explainable AI Module
Why: Users need to understand why the model flagged content as fake.
How: Use Captum to generate attributions and visualize them.
"""

from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from graph_fusion import GraphFakeDetector, build_graph

def explain_prediction(text_feat, img_feat, vid_feat, target=1):
    """
    Generate explanations for the model's prediction.
    Why: Explainability builds trust in AI decisions.
    How: Use Integrated Gradients to find important features.
    """
    try:
        # Build graph
        graph_data = build_graph(text_feat, img_feat, vid_feat)

        # Initialize model
        model = GraphFakeDetector().to('cpu')

        # Initialize Integrated Gradients
        ig = IntegratedGradients(model)

        # Get attributions
        attributions = ig.attribute(
            graph_data.x,
            target=target,
            additional_forward_args=(graph_data.edge_index,)
        )

        return attributions.detach()

    except Exception as e:
        print(f"XAI error: {e}")
        return None

def visualize_image_attribution(attributions, original_image, save_path='explanation.png'):
    """
    Visualize attributions on the image.
    Why: Show which parts of the image influenced the decision.
    How: Create a heatmap overlay on the original image.
    """
    try:
        if attributions is None or original_image is None:
            return

        # Get attribution for image (assuming it's node 1)
        img_attr = attributions[1].cpu().numpy()

        # Reshape to image dimensions (assuming 768 features map to 224x224)
        # This is a simplification; in practice, you'd need proper mapping
        attr_map = img_attr.reshape(1, -1)
        attr_map = cv2.resize(attr_map, (224, 224))

        # Normalize
        attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min())

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attr_map), cv2.COLORMAP_HOT)

        # Overlay on original image
        overlay = cv2.addWeighted(original_image, 0.7, heatmap, 0.3, 0)

        # Save the visualization
        cv2.imwrite(save_path, overlay)
        print(f"Explanation saved to {save_path}")

    except Exception as e:
        print(f"Visualization error: {e}")

def generate_text_explanation(attributions, text_tokens):
    """
    Generate text-based explanation.
    Why: Highlight important words in the text.
    How: Find tokens with high attribution scores.
    """
    try:
        if attributions is None:
            return "No explanation available"

        # Get text attributions (node 0)
        text_attr = attributions[0].cpu().numpy()

        # Find top contributing tokens
        top_indices = np.argsort(text_attr)[-5:][::-1]  # Top 5

        explanation = "Key suspicious elements in text: "
        for idx in top_indices:
            if idx < len(text_tokens):
                explanation += f"'{text_tokens[idx]}' "

        return explanation

    except Exception as e:
        return f"Text explanation error: {e}"

if __name__ == "__main__":
    # Test XAI
    # Create dummy features
    text_feat = torch.randn(1, 768)
    img_feat = torch.randn(1, 768)
    vid_feat = torch.randn(1, 768)

    # Generate explanation
    attributions = explain_prediction(text_feat, img_feat, vid_feat)

    if attributions is not None:
        print("Attributions shape:", attributions.shape)

        # Test visualization (with dummy image)
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        visualize_image_attribution(attributions, dummy_img)

        # Test text explanation
        dummy_tokens = ["breaking", "news", "aliens", "invasion", "detected"]
        text_exp = generate_text_explanation(attributions, dummy_tokens)
        print(text_exp)