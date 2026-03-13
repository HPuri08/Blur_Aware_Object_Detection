def parse_kitti_label(label_file, class_names):
    """
    Parse KITTI format label file and return bounding boxes.
    
    KITTI format:
    class truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y
    
    We only need: class x1 y1 x2 y2
    """
    boxes = []
    
    # Create reverse mapping from class names to IDs
    name_to_id = {name.lower(): id for id, name in class_names.items()}
    
    try:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:  # At least class and bbox coordinates
                    class_name = parts[0].lower()
                    
                    # Map class name to ID
                    if class_name in name_to_id:
                        class_id = name_to_id[class_name]
                        
                        # Extract bounding box coordinates (KITTI: left, top, right, bottom)
                        try:
                            x1, y1, x2, y2 = map(float, parts[4:8])
                            boxes.append([x1, y1, x2, y2, class_id])
                        except ValueError:
                            print(f"Warning: Could not parse coordinates in line: {line.strip()}")
                            continue
                    else:
                        print(f"Warning: Unknown class '{parts[0]}' in {label_file}")
                        
    except FileNotFoundError:
        print(f"Warning: Label file not found: {label_file}")
    except Exception as e:
        print(f"Error parsing {label_file}: {e}")
    
    return boxes

def parse_yolo_label(label_file, class_names, img_width, img_height):
    """
    Alternative parser for YOLO format labels.
    YOLO format: class_id center_x center_y width height (all normalized 0-1)
    """
    boxes = []
    
    try:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        if class_id in class_names:
                            # Convert from normalized YOLO format to absolute coordinates
                            center_x, center_y, width, height = map(float, parts[1:5])
                            
                            # Convert to absolute pixels
                            center_x *= img_width
                            center_y *= img_height
                            width *= img_width
                            height *= img_height
                            
                            # Convert to [x1, y1, x2, y2]
                            x1 = center_x - width / 2
                            y1 = center_y - height / 2
                            x2 = center_x + width / 2
                            y2 = center_y + height / 2
                            
                            boxes.append([x1, y1, x2, y2, class_id])
                    except ValueError:
                        print(f"Warning: Could not parse line: {line.strip()}")
                        continue
                        
    except FileNotFoundError:
        print(f"Warning: Label file not found: {label_file}")
    except Exception as e:
        print(f"Error parsing {label_file}: {e}")
    
    return boxes
