import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as T
from PIL import Image
import os
import math
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionEmbeddingSine(nn.Module):
    """Positional encoding for DETR"""
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        B, C, H, W = x.shape
        mask = torch.zeros((B, H, W), dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class DETR(nn.Module):
    """DETR model with ResNet-50 backbone for COCO"""
    def __init__(self, num_classes=91, num_queries=100, hidden_dim=256):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # ResNet-50 backbone
        backbone = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Reduce channel dimension
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        
        # Positional encoding
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        # Query embeddings
        self.query_embed = nn.Parameter(torch.rand(num_queries, hidden_dim))
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        h = self.input_proj(features)
        pos = self.position_embedding(h)
        batch_size, c, h_dim, w_dim = h.shape
        h = h.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)
        query_embed = self.query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        hs = self.transformer(h + pos, query_embed)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        outputs_class = outputs_class.permute(1, 0, 2)
        outputs_coord = outputs_coord.permute(1, 0, 2)
        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

def inspect_checkpoint(path):
    """Inspect checkpoint structure for debugging"""
    try:
        checkpoint = torch.load(path, map_location='cpu')
        print(f"\nInspecting checkpoint: {path}")
        if isinstance(checkpoint, dict):
            print("Keys:", list(checkpoint.keys()))
            for key in ['model_state_dict', 'state_dict', 'model']:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    print(f"\n{key} contains {len(state_dict)} parameters:")
                    print("First 5 parameter names and shapes:")
                    for k, v in list(state_dict.items())[:5]:
                        print(f"  {k}: {v.shape}")
        else:
            print("Direct state dict")
            print(f"Contains {len(checkpoint)} parameters")
            print("First 5 parameter names and shapes:")
            for k, v in list(checkpoint.items())[:5]:
                print(f"  {k}: {v.shape}")
    except Exception as e:
        print(f"Error inspecting checkpoint: {e}")

def load_detr_model(detr_path):
    """Load pretrained DETR model for COCO"""
    try:
        print(f"Loading DETR model from {detr_path}")
        model = DETR(num_classes=91)
        
        # Inspect checkpoint
        inspect_checkpoint(detr_path)
        
        checkpoint = torch.load(detr_path, map_location=device)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint.get('model') or checkpoint
        else:
            state_dict = checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded DETR weights with strict=False")
        
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading DETR model: {e}")
        print("Creating DETR model with random weights for testing...")
        model = DETR(num_classes=91)
        model.to(device)
        model.eval()
        return model

transform = T.Compose([
    T.Resize((800, 1333)),  # DETR typical input size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_with_detr(model, image_path, confidence_threshold=0.0001):  # Lowered to 0.0001
    """Run inference with DETR model for COCO"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    print(f"\nProcessing {image_path}")
    print(f"Input image shape: {img_tensor.shape}")
    print(f"Original image size: {original_size}")
    
    with torch.no_grad():
        outputs = model(img_tensor)
    
    pred_logits = outputs['pred_logits'][0]
    pred_boxes = outputs['pred_boxes'][0]
    
    print(f"Prediction logits shape: {pred_logits.shape}")
    print(f"Prediction boxes shape: {pred_boxes.shape}")
    
    pred_probs = F.softmax(pred_logits, -1)
    class_probs = pred_probs[:, :-1]  # Exclude no-object class
    max_probs, class_indices = class_probs.max(dim=1)
    print(f"All confidence scores: {max_probs.tolist()}")
    
    predictions = []
    for i in range(len(pred_logits)):
        max_prob = max_probs[i].item()
        class_idx = class_indices[i].item()
        if max_prob > confidence_threshold and class_idx < 91:  # Ensure class index is within 0-90
            x_center, y_center, width, height = pred_boxes[i]
            x_center = x_center * original_size[0]
            y_center = y_center * original_size[1]
            width = width * original_size[0]
            height = height * original_size[1]
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            x1 = max(0, min(original_size[0], x1))
            y1 = max(0, min(original_size[1], y1))
            x2 = max(0, min(original_size[0], x2))
            y2 = max(0, min(original_size[1], y2))
            
            if x2 > x1 and y2 > y1:
                predictions.append([x1, y1, x2, y2, max_prob, class_idx])
    
    print(f"Number of predictions above threshold: {len(predictions)}")
    return predictions

if __name__ == "__main__":
    detr_path = "pretrained_models/detr_r50.pth"
    output_dir = "output_viz/coco"
    test_dir = r"D:\harsh\Saarland University\Sem 2\HLCV\hlcv_project\data\testing\image_2"
    
    # Load model
    model = load_detr_model(detr_path)
    
    # COCO class names (91 classes, 80 objects + background handling)
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]
    
    # Get all PNG images and limit to first 5
    images = sorted(glob.glob(os.path.join(test_dir, "*.png")))[:5]
    print(f"Processing the first {len(images)} images")
    
    for image_path in images:
        # Run inference
        predictions = predict_with_detr(model, image_path, confidence_threshold=0.0001)
        
        # Load image for visualization
        image = Image.open(image_path).convert('RGB')
        
        # Create figure
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))
        
        # Left: Original
        axs[0].imshow(image)
        axs[0].axis('off')
        axs[0].set_title('Original Image')
        
        # Right: With boxes
        axs[1].imshow(image)
        axs[1].axis('off')
        axs[1].set_title('Image with Bounding Boxes')
        
        for x1, y1, x2, y2, conf, cls in predictions:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
            axs[1].add_patch(rect)
            label = f'{coco_classes[cls]}: {conf:.2f}'
            axs[1].text(x1, y1 - 5, label, bbox=dict(facecolor='red', alpha=0.5), fontsize=8, color='white')
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, os.path.basename(image_path))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved visualization for {image_path} to {save_path}")