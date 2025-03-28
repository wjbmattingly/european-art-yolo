from datasets import load_dataset
import os
import shutil
from sklearn.model_selection import train_test_split
import yaml
from PIL import Image
import json
from tqdm import tqdm
from collections import defaultdict

class CategoryTracker:
    def __init__(self):
        self.name_to_idx = {}  # Maps category name to YOLO index
        self.names = []        # List of category names in order
        self.total_counts = defaultdict(int)    # Counts by category name
        self.train_counts = defaultdict(int)    # Counts in train set
        self.val_counts = defaultdict(int)      # Counts in val set
    
    def get_class_id(self, category_name):
        if category_name not in self.name_to_idx:
            self.name_to_idx[category_name] = len(self.names)
            self.names.append(category_name)
        return self.name_to_idx[category_name]
    
    def update_counts(self, category_name, split):
        self.total_counts[category_name] += 1
        if split == 'train':
            self.train_counts[category_name] += 1
        else:
            self.val_counts[category_name] += 1

def resize_image_and_boxes(image, boxes, max_size=1000):
    """Resize image and adjust bounding boxes while maintaining aspect ratio"""
    # Get original dimensions
    width, height = image.size
    
    # Calculate scaling factor
    scale = min(max_size/width, max_size/height)
    
    # Only resize if image is larger than max_size
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Adjust boxes
        boxes = [[x * scale, y * scale, w * scale, h * scale] for x, y, w, h in boxes]
    
    return image, boxes, image.size[0], image.size[1]

def convert_to_yolo(example, output_dir, split, category_tracker):
    """Convert a single example to YOLO format and save files"""
    # Create directories if they don't exist
    images_dir = os.path.join(output_dir, split, 'images')
    labels_dir = os.path.join(output_dir, split, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Parse the JSON string
    annotations_data = json.loads(example['annotations'])
    
    # Get image info and boxes
    image_info = annotations_data['images'][0]
    image_id = image_info['id']
    image = example['image']
    width = image_info['width']
    height = image_info['height']
    
    # Resize image and adjust boxes
    boxes = [ann['bbox'] for ann in annotations_data['annotations']]
    image, boxes, width, height = resize_image_and_boxes(image, boxes)
    
    # Save resized image
    image.save(os.path.join(images_dir, f"{image_id:08d}.jpg"))
    
    # Convert and save annotations
    label_file = os.path.join(labels_dir, f"{image_id:08d}.txt")
    with open(label_file, 'w') as f:
        for ann, (x, y, w, h) in zip(annotations_data['annotations'], boxes):
            # Get category name and YOLO class ID
            category_name = next(cat['name'] for cat in annotations_data['categories'] 
                               if cat['id'] == ann['category_id'])
            class_id = category_tracker.get_class_id(category_name)
            category_tracker.update_counts(category_name, split)
            
            # Convert to normalized coordinates
            x_center = (x + w/2) / width
            y_center = (y + h/2) / height
            w = w / width
            h = h / height
            
            f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

def create_dataset(output_dir, train_split=0.8):
    """Create YOLO dataset with train/val split"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("wjbmattingly/european-art")
    
    # Initialize category tracker
    category_tracker = CategoryTracker()
    
    # Split dataset
    train_idx, val_idx = train_test_split(
        range(len(dataset['train'])),
        train_size=train_split,
        random_state=42
    )
    
    # Convert and save examples
    for idx in tqdm(train_idx, desc="Converting train examples"):
        convert_to_yolo(dataset['train'][idx], output_dir, 'train', category_tracker)
    
    for idx in tqdm(val_idx, desc="Converting val examples"):
        convert_to_yolo(dataset['train'][idx], output_dir, 'val', category_tracker)
    
    # Create data.yaml with counts and absolute paths
    data = {
        'path': os.path.abspath(output_dir),  # Use absolute path
        'train': os.path.join(os.path.abspath(output_dir), 'train', 'images'),  # Absolute path to train
        'val': os.path.join(os.path.abspath(output_dir), 'val', 'images'),  # Absolute path to val
        'nc': len(category_tracker.names),
        'names': category_tracker.names,
        'counts': {
            'total': dict(category_tracker.total_counts),
            'train': dict(category_tracker.train_counts),
            'val': dict(category_tracker.val_counts)
        }
    }
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
        
    # Print summary
    print("\nDataset Summary:")
    print(f"Dataset created in: {os.path.abspath(output_dir)}")
    print(f"Total number of classes: {len(category_tracker.names)}")
    print("\nClass distribution:")
    for name in category_tracker.names:
        print(f"\n{name}:")
        print(f"  Total: {category_tracker.total_counts[name]}")
        print(f"  Train: {category_tracker.train_counts[name]}")
        print(f"  Val: {category_tracker.val_counts[name]}")

if __name__ == "__main__":
    # Specify your output directory here
    output_dir = "european_art_dataset"
    create_dataset(output_dir, 0.8)

