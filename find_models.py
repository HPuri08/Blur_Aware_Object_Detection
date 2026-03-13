import os
from pathlib import Path
import torch

def find_all_model_files():
    """Find all model files in the project"""
    print("=== Starting model search ===")
    print(f"Searching from directory: {os.getcwd()}")
    
    # Look for all possible model file extensions
    extensions = ['.pth', '.pt', '.pkl', '.ckpt']
    model_files = []
    
    # Search recursively
    for root, dirs, files in os.walk('.'):
        print(f"Checking directory: {root}")
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, file)
                try:
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    model_files.append((full_path, size_mb))
                    print(f"  Found: {full_path} ({size_mb:.1f} MB)")
                except Exception as e:
                    print(f"  Error checking {full_path}: {e}")
    
    if not model_files:
        print("❌ No model files found!")
        return []
    
    print(f"\n=== Summary: Found {len(model_files)} model files ===")
    
    # Sort by size
    model_files.sort(key=lambda x: x[1], reverse=True)
    
    for filepath, size in model_files:
        print(f"{filepath:50s} {size:8.1f} MB")
    
    return model_files

def inspect_model_files(model_files):
    """Inspect the contents of model files"""
    print(f"\n=== Inspecting model files ===")
    
    for filepath, size in model_files:
        print(f"\n--- {filepath} ({size:.1f} MB) ---")
        
        try:
            # Load without moving to GPU
            checkpoint = torch.load(filepath, map_location='cpu')
            print(f"✅ Loaded successfully")
            print(f"Type: {type(checkpoint)}")
            
            if isinstance(checkpoint, dict):
                keys = list(checkpoint.keys())
                print(f"Dictionary with {len(keys)} keys: {keys}")
                
                # Look for model weights
                model_keys = ['model', 'model_state_dict', 'state_dict', 'net']
                for key in model_keys:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        if hasattr(state_dict, 'keys'):
                            param_names = list(state_dict.keys())
                            print(f"  {key}: {len(param_names)} parameters")
                            
                            # Check for DETR-specific components
                            detr_components = []
                            backbone_components = []
                            
                            for name in param_names:
                                if any(comp in name.lower() for comp in 
                                      ['transformer', 'query_embed', 'class_embed', 'bbox_embed']):
                                    detr_components.append(name)
                                elif any(comp in name.lower() for comp in 
                                        ['backbone', 'conv1', 'layer1', 'layer2', 'layer3', 'layer4']):
                                    backbone_components.append(name)
                            
                            if detr_components:
                                print(f"    🎯 DETR components: {len(detr_components)}")
                                for comp in detr_components[:3]:
                                    print(f"      {comp}")
                                if len(detr_components) > 3:
                                    print(f"      ... and {len(detr_components)-3} more")
                                    
                            if backbone_components:
                                print(f"    🏗️  Backbone components: {len(backbone_components)}")
                                for comp in backbone_components[:3]:
                                    print(f"      {comp}")
                                if len(backbone_components) > 3:
                                    print(f"      ... and {len(backbone_components)-3} more")
                            
                            if not detr_components and not backbone_components:
                                print(f"    📝 First few parameters:")
                                for name in param_names[:5]:
                                    print(f"      {name}")
                
                # Check for training metadata
                meta_keys = ['epoch', 'loss', 'accuracy', 'best_acc', 'optimizer', 'lr_scheduler']
                meta_info = {k: v for k, v in checkpoint.items() if k in meta_keys}
                if meta_info:
                    print(f"  📊 Training info: {meta_info}")
                    
            else:
                # Direct state dict
                if hasattr(checkpoint, 'keys'):
                    param_names = list(checkpoint.keys())
                    print(f"Direct state dict with {len(param_names)} parameters")
                    for name in param_names[:5]:
                        print(f"  {name}")
                else:
                    print(f"Unknown format: {type(checkpoint)}")
                    
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")

def main():
    print("🔍 DETR Model Finder")
    print("=" * 50)
    
    # Find all model files
    model_files = find_all_model_files()
    
    if model_files:
        # Inspect them
        inspect_model_files(model_files)
        
        # Recommendations
        print(f"\n=== Recommendations ===")
        large_files = [f for f in model_files if f[1] > 50]
        if large_files:
            print(f"🎯 Large files (likely complete models):")
            for filepath, size in large_files:
                print(f"  {filepath} ({size:.1f} MB)")
        else:
            print(f"⚠️  No large model files found.")
            print(f"   Your ResNet-18 files are just backbones, not complete DETR models.")
            print(f"   You may need to train a complete DETR model first.")
    
    else:
        print("❌ No model files found in the project!")
        print("   Make sure you're running this from the correct directory.")

if __name__ == "__main__":
    main()