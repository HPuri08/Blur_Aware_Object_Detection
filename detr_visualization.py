import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import numpy as np
import torchvision.models as models
from plot_utils import plot_gt_vs_pred
from label_parser import parse_kitti_label
import math

# ----------------------------
# Config
# ----------------------------
datasets = {
    "clean": {
        "img_dir": Path("split_dataset/clear/images/val"),
        "label_dir": Path("split_dataset/labels/val"),
        "out_dir": Path("output_viz/clean"),
        "backbone_path": Path("pretrained_models/resnet18_simclr_kitti_clean.pth"),
        "detr_path": Path("pretrained_models/detr_r50.pth")  # Add your DETR path here
    }
    # "blurred": {
    #     "img_dir": Path("data/test_blurred/images"),
    #     "label_dir": Path("data/test_blurred/labels"),
    #     "out_dir": Path("output_viz/blurred"),
    #     "backbone_path": Path("pretrained_models/resnet18_simclr_kitti_blurred.pth"),
    #     "detr_path": Path("pretrained_models/detr_r50.pth")  # Add your DETR path here
    # }
}

class_names = {
    0: "Car",
    1: "Pedestrian", 
    2: "Cyclist"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# DETR Model Implementation
# ----------------------------
class PositionEmbeddingSine(torch.nn.Module):
    """
    Positional encoding for DETR
    """
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

class DETR(torch.nn.Module):
    """
    DETR model with ResNet-50 backbone
    """
    def __init__(self, num_classes, num_queries=100, hidden_dim=256):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # ResNet-50 backbone
        backbone = models.resnet50(pretrained=False)
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        
        # Reduce channel dimension
        self.input_proj = torch.nn.Conv2d(2048, hidden_dim, kernel_size=1)
        
        # Positional encoding
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        
        # Transformer
        self.transformer = torch.nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        # Query embeddings
        self.query_embed = torch.nn.Parameter(torch.rand(num_queries, hidden_dim))
        
        # Prediction heads
        self.class_embed = torch.nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 4)
        )
        
    def forward(self, x):
        # Backbone
        features = self.backbone(x)
        
        # Project to hidden dimension
        h = self.input_proj(features)
        
        # Positional encoding
        pos = self.position_embedding(h)
        
        # Flatten for transformer
        batch_size, c, h_dim, w_dim = h.shape
        h = h.flatten(2).permute(2, 0, 1)  # (H*W, batch_size, hidden_dim)
        pos = pos.flatten(2).permute(2, 0, 1)  # (H*W, batch_size, hidden_dim)
        
        # Query embeddings
        query_embed = self.query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        
        # Transformer
        hs = self.transformer(h + pos, query_embed)[0]  # (num_queries, batch_size, hidden_dim)
        
        # Predictions
        outputs_class = self.class_embed(hs)  # (num_queries, batch_size, num_classes + 1)
        outputs_coord = self.bbox_embed(hs).sigmoid()  # (num_queries, batch_size, 4)
        
        # Reshape outputs
        outputs_class = outputs_class.permute(1, 0, 2)  # (batch_size, num_queries, num_classes + 1)
        outputs_coord = outputs_coord.permute(1, 0, 2)  # (batch_size, num_queries, 4)
        
        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

def load_detr_model(detr_path, num_classes):
    """Load DETR model with proper error handling"""
    try:
        print(f"Loading DETR model from {detr_path}")
        
        # Create model
        model = DETR(num_classes)
        
        # Load checkpoint
        checkpoint = torch.load(detr_path, map_location=device)
        
        # Debug: Check what's in the checkpoint
        print("Checkpoint keys:", list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Direct state dict")
        
        # Try different ways to load the state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("Loading from model_state_dict")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("Loading from state_dict")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("Loading from model key")
            else:
                state_dict = checkpoint
                print("Using checkpoint as state dict")
        else:
            state_dict = checkpoint
            print("Direct state dict")
        
        # Load state dict with error handling
        try:
            model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded DETR weights!")
        except Exception as e:
            print(f"Error loading state dict: {e}")
            print("Loading with strict=False and filtering compatible keys...")
            
            model_dict = model.state_dict()
            filtered_dict = {}
            
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                else:
                    print(f"Skipping key {k}: {'shape mismatch' if k in model_dict else 'not found'}")
            
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            print(f"Loaded {len(filtered_dict)} compatible parameters")
        
        model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error loading DETR model: {e}")
        print("Creating DETR model with random weights for testing...")
        model = DETR(num_classes)
        model.to(device)
        model.eval()
        return model

def inspect_detr_checkpoint(detr_path):
    """Inspect DETR checkpoint structure"""
    try:
        checkpoint = torch.load(detr_path, map_location='cpu')
        print(f"\n=== Inspecting DETR checkpoint: {detr_path} ===")
        print(f"Type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print("Keys:", list(checkpoint.keys()))
            
            # Look for model weights
            for key in ['model_state_dict', 'state_dict', 'model', 'net']:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    print(f"\n{key} contains {len(state_dict)} parameters:")
                    
                    # Group parameters by component
                    components = {}
                    for param_name in state_dict.keys():
                        component = param_name.split('.')[0]
                        if component not in components:
                            components[component] = 0
                        components[component] += 1
                    
                    print("Parameter groups:")
                    for comp, count in sorted(components.items()):
                        print(f"  {comp}: {count} parameters")
                    
                    # Show some example parameter names
                    print("\nFirst 15 parameter names:")
                    for i, param_name in enumerate(list(state_dict.keys())[:15]):
                        print(f"  {param_name}: {state_dict[param_name].shape}")
                    
                    break
            
            # Check for training info
            for info_key in ['epoch', 'loss', 'best_loss', 'optimizer']:
                if info_key in checkpoint:
                    if info_key == 'optimizer':
                        print(f"{info_key}: Present")
                    else:
                        print(f"{info_key}: {checkpoint[info_key]}")
        else:
            print("Direct state dict")
            print(f"Contains {len(checkpoint)} parameters")
            
    except Exception as e:
        print(f"Error inspecting DETR checkpoint: {e}")

# Image preprocessing for DETR
transform = T.Compose([
    T.Resize((800, 1333)),  # DETR typical input size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_with_detr(model, image_path, confidence_threshold=0.7):
    """
    Run inference with DETR model
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Transform image
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    print(f"Input image shape: {img_tensor.shape}")
    print(f"Original image size: {original_size}")
    
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Extract predictions
    pred_logits = outputs['pred_logits'][0]  # Remove batch dimension
    pred_boxes = outputs['pred_boxes'][0]    # Remove batch dimension
    
    print(f"Prediction logits shape: {pred_logits.shape}")
    print(f"Prediction boxes shape: {pred_boxes.shape}")
    
    # Convert logits to probabilities
    pred_probs = F.softmax(pred_logits, -1)
    
    # Get predictions above confidence threshold
    predictions = []
    
    for i in range(len(pred_logits)):
        # Get class probabilities (excluding no-object class)
        class_probs = pred_probs[i, :-1]  # Exclude last class (no-object)
        max_prob, class_idx = class_probs.max(0)
        
        if max_prob > confidence_threshold:
            # Convert normalized coordinates to absolute coordinates
            x_center, y_center, width, height = pred_boxes[i]
            
            # Convert to absolute coordinates
            x_center = x_center * original_size[0]
            y_center = y_center * original_size[1]
            width = width * original_size[0]
            height = height * original_size[1]
            
            # Convert to [x1, y1, x2, y2] format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Clip to image boundaries
            x1 = max(0, min(original_size[0], x1))
            y1 = max(0, min(original_size[1], y1))
            x2 = max(0, min(original_size[0], x2))
            y2 = max(0, min(original_size[1], y2))
            
            if x2 > x1 and y2 > y1:  # Valid box
                predictions.append([x1, y1, x2, y2, max_prob.item(), class_idx.item()])
    
    return predictions

def visualize_single_image(dataset_tag, image_name=None):
    """
    Visualize DETR predictions vs ground truth on a single image
    """
    if dataset_tag not in datasets:
        print(f"Unknown dataset tag: {dataset_tag}")
        return
    
    paths = datasets[dataset_tag]
    img_dir = paths["img_dir"]
    label_dir = paths["label_dir"]
    out_dir = paths["out_dir"]
    detr_path = paths["detr_path"]
    
    print(f"\n=== Processing {dataset_tag} dataset with DETR ===")
    
    # Check if paths exist
    if not detr_path.exists():
        print(f"DETR model file not found: {detr_path}")
        print("Please update the detr_path in the config section")
        return
        
    if not img_dir.exists():
        print(f"Image directory not found: {img_dir}")
        return
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find image file
    if image_name:
        img_file = img_dir / f"{image_name}.png"
        if not img_file.exists():
            img_file = img_dir / f"{image_name}.jpg"
        if not img_file.exists():
            print(f"Image not found: {image_name}")
            return
    else:
        # Get first image
        image_files = sorted(img_dir.glob("*.png"))
        if not image_files:
            image_files = sorted(img_dir.glob("*.jpg"))
        if not image_files:
            print("No images found in directory")
            return
        img_file = image_files[0]
    
    print(f"Processing image: {img_file.name}")
    
    # Check corresponding label file
    label_file = label_dir / f"{img_file.stem}.txt"
    if not label_file.exists():
        print(f"Label file not found: {label_file}")
        return
    
    # Inspect DETR checkpoint
    inspect_detr_checkpoint(detr_path)
    
    # Load DETR model
    print(f"\nLoading DETR model...")
    model = load_detr_model(detr_path, len(class_names))
    
    # Parse ground truth
    gt_boxes = parse_kitti_label(label_file, class_names)
    print(f"\nFound {len(gt_boxes)} ground truth boxes:")
    for i, box in enumerate(gt_boxes):
        x1, y1, x2, y2, cls_id = box
        print(f"  GT {i+1}: {class_names[cls_id]} at ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
    
    # Get DETR predictions
    print(f"\nRunning DETR inference...")
    pred_boxes = predict_with_detr(model, img_file, confidence_threshold=0.5)
    print(f"\nFound {len(pred_boxes)} DETR predictions:")
    for i, box in enumerate(pred_boxes):
        x1, y1, x2, y2, conf, cls_id = box
        print(f"  Pred {i+1}: {class_names[cls_id]} ({conf:.3f}) at ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
    
    # Save visualizations
    save_path = out_dir / f"{img_file.stem}_detr_predictions.png"
    plot_gt_vs_pred(img_file, gt_boxes, pred_boxes, class_names, save_path=save_path)
    print(f"\nDETR visualization saved to: {save_path}")

def main():
    """Main function for DETR visualization"""
    print("=== DETR Object Detection Visualization ===")
    
    # Choose which dataset to visualize
    dataset_to_use = "clean"  # Change to "blurred" if needed
    specific_image = None     # Set to image name without extension, or None for first image
    
    visualize_single_image(dataset_to_use, specific_image)

if __name__ == "__main__":
    main()