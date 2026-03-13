import torch

# Check your model files
for model_path in ["pretrained_models/resnet18_simclr_kitti_clean.pth", 
                   "pretrained_models/resnet18_simclr_kitti_blurred.pth"]:
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"\n=== {model_path} ===")
        print("Type:", type(checkpoint))
        if isinstance(checkpoint, dict):
            print("Keys:", list(checkpoint.keys()))
        print("First few parameter names:")
        if hasattr(checkpoint, 'keys'):
            for i, key in enumerate(list(checkpoint.keys())[:5]):
                print(f"  {key}")
    except Exception as e:
        print(f"Error loading {model_path}: {e}")