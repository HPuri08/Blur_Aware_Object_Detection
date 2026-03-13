import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_gt_vs_pred(image_path, gt_boxes, pred_boxes, class_names, save_path=None):
    """
    Plot ground truth vs predictions with improved error handling and visualization
    """
    # Load and convert image
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    img_height, img_width = img.shape[:2]
    print(f"Image size: {img_width} x {img_height}")
    
    # Draw ground truth boxes (in blue/red)
    for i, box in enumerate(gt_boxes):
        try:
            x1, y1, x2, y2, cls_id = box
            x1, y1, x2, y2, cls_id = int(x1), int(y1), int(x2), int(y2), int(cls_id)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(img_width - 1, x1))
            y1 = max(0, min(img_height - 1, y1))
            x2 = max(x1 + 1, min(img_width, x2))
            y2 = max(y1 + 1, min(img_height, y2))
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red for GT
            
            # Draw label
            label = f"GT: {class_names.get(cls_id, f'Class_{cls_id}')}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 0, 255), -1)
            
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
            print(f"Drew GT box {i+1}: {class_names.get(cls_id, f'Class_{cls_id}')} at ({x1}, {y1}, {x2}, {y2})")
            
        except Exception as e:
            print(f"Error drawing GT box {i}: {e}")
            continue
    
    # Draw prediction boxes (in green)
    for i, box in enumerate(pred_boxes):
        try:
            if len(box) == 6:  # [x1, y1, x2, y2, conf, cls_id]
                x1, y1, x2, y2, conf, cls_id = box
            elif len(box) == 5:  # [x1, y1, x2, y2, cls_id] - assume conf = 1.0
                x1, y1, x2, y2, cls_id = box
                conf = 1.0
            else:
                print(f"Warning: Unexpected prediction box format: {box}")
                continue
                
            x1, y1, x2, y2, cls_id = int(x1), int(y1), int(x2), int(y2), int(cls_id)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(img_width - 1, x1))
            y1 = max(0, min(img_height - 1, y1))
            x2 = max(x1 + 1, min(img_width, x2))
            y2 = max(y1 + 1, min(img_height, y2))
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green for predictions
            
            # Draw label
            label = f"PRED: {class_names.get(cls_id, f'Class_{cls_id}')} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(img, (x1, y2 + 5), 
                         (x1 + label_size[0], y2 + label_size[1] + 15), (0, 255, 0), -1)
            
            cv2.putText(img, label, (x1, y2 + label_size[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                       
            print(f"Drew prediction box {i+1}: {class_names.get(cls_id, f'Class_{cls_id}')} ({conf:.3f}) at ({x1}, {y1}, {x2}, {y2})")
            
        except Exception as e:
            print(f"Error drawing prediction box {i}: {e}")
            continue
    
    # Add legend
    legend_height = 80 if len(pred_boxes) > 0 else 55
    legend_y = 30
    cv2.rectangle(img, (10, 10), (200, legend_height), (255, 255, 255), -1)
    cv2.rectangle(img, (10, 10), (200, legend_height), (0, 0, 0), 2)
    cv2.putText(img, "Red: Ground Truth", (15, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    if len(pred_boxes) > 0:
        cv2.putText(img, "Green: Predictions", (15, legend_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(img, "No Predictions", (15, legend_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
    
    # Save or display
    if save_path:
        try:
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Ground Truth vs Predictions\nGT boxes: {len(gt_boxes)}, Pred boxes: {len(pred_boxes)}')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Visualization saved to: {save_path}")
        except Exception as e:
            print(f"Error saving visualization: {e}")
    else:
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Ground Truth vs Predictions\nGT boxes: {len(gt_boxes)}, Pred boxes: {len(pred_boxes)}')
        plt.show()

def plot_image_with_boxes(image_path, boxes, class_names, box_type="Boxes", save_path=None):
    """
    Plot image with only one type of boxes (either GT or predictions)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    img_height, img_width = img.shape[:2]
    
    # Choose color based on box type
    color = (255, 0, 0) if "GT" in box_type else (0, 255, 0)
    
    for i, box in enumerate(boxes):
        try:
            if len(box) == 5:  # [x1, y1, x2, y2, cls_id]
                x1, y1, x2, y2, cls_id = box
                conf = 1.0
            elif len(box) == 6:  # [x1, y1, x2, y2, conf, cls_id]
                x1, y1, x2, y2, conf, cls_id = box
            else:
                continue
                
            x1, y1, x2, y2, cls_id = int(x1), int(y1), int(x2), int(y2), int(cls_id)
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_names.get(cls_id, f'Class_{cls_id}')}"
            if len(box) == 6:
                label += f" {conf:.2f}"
                
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
        except Exception as e:
            print(f"Error drawing box {i}: {e}")
            continue
    
    if save_path:
        plt.imsave(save_path, img)
    else:
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'{box_type} - {len(boxes)} boxes')
        plt.show()