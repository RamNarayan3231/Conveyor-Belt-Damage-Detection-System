import os
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import cv2
from tqdm import tqdm


class ConveyorBeltTrainer:
    """Trainer for conveyor belt damage detection model."""
    
    def __init__(self, data_yaml_path: str, model_name: str = 'yolov8n.pt'):
        """
        Initialize trainer.
        
        Args:
            data_yaml_path: Path to dataset YAML configuration
            model_name: Base YOLO model to use
        """
        self.data_yaml = data_yaml_path
        self.model_name = model_name
        self.model = None
        self.results = None
    
    def create_data_yaml(self, train_path: str, val_path: str, 
                         test_path: str, class_names: list, output_path: str = 'dataset.yaml'):
        """
        Create YAML configuration file for dataset.
        
        Args:
            train_path: Path to training images
            val_path: Path to validation images
            test_path: Path to test images
            class_names: List of class names
            output_path: Output YAML file path
        """
        data_config = {
            'train': train_path,
            'val': val_path,
            'test': test_path,
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"Created data configuration at: {output_path}")
        return output_path
    
    def train(self, epochs: int = 100, imgsz: int = 640, batch_size: int = 16,
              device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
              project: str = 'runs/train', name: str = 'conveyor_belt_detection'):
        """
        Train the YOLO model.
        
        Args:
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Batch size
            device: Device to train on ('cuda' or 'cpu')
            project: Project directory for saving results
            name: Name of the training run
        """
        print(f"Training on device: {device}")
        
        # Initialize model
        self.model = YOLO(self.model_name)
        
        # Train the model
        self.results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=project,
            name=name,
            patience=10,
            save=True,
            save_period=10,
            cache=True,
            workers=4,  # Reduced workers for Windows
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0
        )
        
        print("Training completed!")
        return self.results
    
    def evaluate(self, model_path: str, data_yaml: str = None):
        """
        Evaluate the trained model.
        
        Args:
            model_path: Path to trained model weights
            data_yaml: Path to dataset YAML (uses self.data_yaml if None)
        """
        model = YOLO(model_path)
        eval_data = data_yaml or self.data_yaml
        
        metrics = model.val(data=eval_data)
        
        print("\n=== Evaluation Results ===")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP75: {metrics.box.map75:.4f}")
        
        return metrics
    
    def plot_training_curves(self, save_path: str = 'training_curves.png'):
        """Plot training metrics."""
        if self.results is None:
            print("No training results available. Run train() first.")
            return
        
        # Extract metrics
        metrics = self.results.results
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot losses
        epochs = range(1, len(metrics['train/box_loss']) + 1)
        
        axes[0, 0].plot(epochs, metrics['train/box_loss'], label='Train Box Loss')
        axes[0, 0].plot(epochs, metrics['val/box_loss'], label='Val Box Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Box Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('Box Loss')
        
        axes[0, 1].plot(epochs, metrics['train/cls_loss'], label='Train Cls Loss')
        axes[0, 1].plot(epochs, metrics['val/cls_loss'], label='Val Cls Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Class Loss')
        axes[0, 1].legend()
        axes[0, 1].set_title('Class Loss')
        
        axes[0, 2].plot(epochs, metrics['train/dfl_loss'], label='Train DFL Loss')
        axes[0, 2].plot(epochs, metrics['val/dfl_loss'], label='Val DFL Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('DFL Loss')
        axes[0, 2].legend()
        axes[0, 2].set_title('DFL Loss')
        
        # Plot metrics
        axes[1, 0].plot(epochs, metrics['metrics/mAP50-95(B)'], label='mAP50-95')
        axes[1, 0].plot(epochs, metrics['metrics/mAP50(B)'], label='mAP50')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].legend()
        axes[1, 0].set_title('mAP Metrics')
        
        axes[1, 1].plot(epochs, metrics['metrics/precision(B)'], label='Precision')
        axes[1, 1].plot(epochs, metrics['metrics/recall(B)'], label='Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].set_title('Precision & Recall')
        
        # Remove unused subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Training curves saved to: {save_path}")


def prepare_dataset_from_yolo_format(dataset_dir: str, output_dir: str = 'prepared_dataset'):
    """
    Prepare dataset from YOLO format for training.
    
    Args:
        dataset_dir: Directory containing images and labels in YOLO format
        output_dir: Output directory for prepared dataset
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Find all image files (support multiple extensions)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(dataset_path.glob(ext))
    
    # Also check if images are in a subdirectory
    if len(image_files) == 0:
        # Check for images subdirectory
        images_dir = dataset_path / 'images'
        if images_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(images_dir.glob(ext))
    
    # Also check for images in current directory with different naming
    if len(image_files) == 0:
        # Try to find any image files recursively
        image_files = list(dataset_path.rglob('*.jpg')) + list(dataset_path.rglob('*.png')) + \
                     list(dataset_path.rglob('*.jpeg'))
    
    print(f"Found {len(image_files)} image files")
    
    if len(image_files) == 0:
        print("\n❌ No images found! Please check your dataset directory.")
        print(f"Looking in: {dataset_path.absolute()}")
        print("\nYour dataset should contain images and corresponding .txt label files.")
        print("Example structure:")
        print("  dataset/")
        print("  ├── image1.jpg")
        print("  ├── image1.txt")
        print("  ├── image2.jpg")
        print("  └── image2.txt")
        return None
    
    # Filter images that have corresponding label files
    valid_images = []
    for img_file in image_files:
        label_file = img_file.with_suffix('.txt')
        if label_file.exists():
            valid_images.append(img_file)
        else:
            print(f"Warning: No label file found for {img_file.name}")
    
    if len(valid_images) == 0:
        print("\n❌ No images with corresponding label files found!")
        print("Each image should have a .txt file with the same name containing YOLO annotations.")
        return None
    
    print(f"Found {len(valid_images)} images with valid label files")
    
    # Split dataset
    train_files, temp_files = train_test_split(valid_images, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val: {len(val_files)} images")
    print(f"  Test: {len(test_files)} images")
    
    # Copy files
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for img_file in tqdm(files, desc=f"Copying {split} files"):
            # Copy image
            dest_img = output_path / split / 'images' / img_file.name
            shutil.copy2(img_file, dest_img)
            
            # Copy corresponding label file
            label_file = img_file.with_suffix('.txt')
            if label_file.exists():
                dest_label = output_path / split / 'labels' / label_file.name
                shutil.copy2(label_file, dest_label)
    
    print(f"\n✅ Dataset prepared at: {output_path}")
    return output_path


def create_sample_annotations_for_testing(output_dir: str = 'sample_dataset'):
    """
    Create sample annotations for testing the pipeline
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create a sample image (black canvas)
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Draw a simulated scratch (white line)
    cv2.line(img, (100, 320), (540, 320), (255, 255, 255), 3)
    
    # Draw simulated edge damage (red box near edge)
    cv2.rectangle(img, (20, 20), (100, 100), (0, 0, 255), 2)
    
    # Save image
    img_path = output_path / 'sample_image.jpg'
    cv2.imwrite(str(img_path), img)
    
    # Create YOLO format label
    # Format: class_id x_center y_center width height (normalized)
    with open(output_path / 'sample_image.txt', 'w') as f:
        # Scratch annotation (class 0)
        f.write("0 0.5 0.5 0.6 0.05\n")
        # Edge damage annotation (class 1)
        f.write("1 0.1 0.1 0.15 0.15\n")
    
    print(f"Created sample dataset at: {output_path}")
    return output_path


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Conveyor Belt Damage Detection Model')
    parser.add_argument('--dataset_dir', type=str, required=False,
                       help='Directory containing YOLO format dataset')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Base YOLO model (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt)')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample dataset for testing')
    
    args = parser.parse_args()
    
    # Create sample dataset if requested or no dataset provided
    if args.create_sample or not args.dataset_dir:
        print("Creating sample dataset for testing...")
        dataset_dir = create_sample_annotations_for_testing('sample_dataset')
        args.dataset_dir = str(dataset_dir)
        print(f"Using sample dataset at: {args.dataset_dir}")
    
    # Prepare dataset
    print(f"\nPreparing dataset from: {args.dataset_dir}")
    prepared_dataset = prepare_dataset_from_yolo_format(args.dataset_dir)
    
    if prepared_dataset is None:
        print("\n❌ Failed to prepare dataset. Please check your dataset structure.")
        print("\nExpected structure:")
        print("  dataset/")
        print("  ├── image1.jpg")
        print("  ├── image1.txt (YOLO format)")
        print("  ├── image2.jpg")
        print("  └── image2.txt")
        print("\nYOLO label format: class_id x_center y_center width height")
        print("Example: 0 0.5 0.5 0.3 0.1")
        return
    
    # Create data YAML
    trainer = ConveyorBeltTrainer(data_yaml_path='dataset.yaml')
    trainer.create_data_yaml(
        train_path=str(prepared_dataset / 'train' / 'images'),
        val_path=str(prepared_dataset / 'val' / 'images'),
        test_path=str(prepared_dataset / 'test' / 'images'),
        class_names=['scratch', 'edge_damage'],
        output_path='dataset.yaml'
    )
    
    # Train model
    print("\n🚀 Starting training...")
    trainer.train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        project='runs/train',
        name='conveyor_belt_detection'
    )
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Evaluate best model
    best_model_path = 'runs/train/conveyor_belt_detection/weights/best.pt'
    if Path(best_model_path).exists():
        trainer.evaluate(best_model_path)
        
        # Export to ONNX and TorchScript
        model = YOLO(best_model_path)
        model.export(format='onnx', imgsz=args.imgsz)
        model.export(format='torchscript', imgsz=args.imgsz)
        
        # Copy to backend/models directory
        backend_models_dir = Path('backend/models')
        backend_models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_model_path, backend_models_dir / 'best.pt')
        print(f"\n✅ Model copied to: {backend_models_dir / 'best.pt'}")
    
    print("\n✅ Training pipeline completed!")


if __name__ == "__main__":
    main()



# import os
# import yaml
# import torch
# from pathlib import Path
# from ultralytics import YOLO
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# import numpy as np


# class ConveyorBeltTrainer:
#     """Trainer for conveyor belt damage detection model."""
    
#     def __init__(self, data_yaml_path: str, model_name: str = 'yolov8n.pt'):
#         """
#         Initialize trainer.
        
#         Args:
#             data_yaml_path: Path to dataset YAML configuration
#             model_name: Base YOLO model to use
#         """
#         self.data_yaml = data_yaml_path
#         self.model_name = model_name
#         self.model = None
#         self.results = None
    
#     def create_data_yaml(self, train_path: str, val_path: str, 
#                          test_path: str, class_names: list, output_path: str = 'dataset.yaml'):
#         """
#         Create YAML configuration file for dataset.
        
#         Args:
#             train_path: Path to training images
#             val_path: Path to validation images
#             test_path: Path to test images
#             class_names: List of class names
#             output_path: Output YAML file path
#         """
#         data_config = {
#             'train': train_path,
#             'val': val_path,
#             'test': test_path,
#             'nc': len(class_names),
#             'names': class_names
#         }
        
#         with open(output_path, 'w') as f:
#             yaml.dump(data_config, f, default_flow_style=False)
        
#         print(f"Created data configuration at: {output_path}")
#         return output_path
    
#     def train(self, epochs: int = 100, imgsz: int = 640, batch_size: int = 16,
#               device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
#               project: str = 'runs/train', name: str = 'conveyor_belt_detection'):
#         """
#         Train the YOLO model.
        
#         Args:
#             epochs: Number of training epochs
#             imgsz: Image size for training
#             batch_size: Batch size
#             device: Device to train on ('cuda' or 'cpu')
#             project: Project directory for saving results
#             name: Name of the training run
#         """
#         print(f"Training on device: {device}")
        
#         # Initialize model
#         self.model = YOLO(self.model_name)
        
#         # Train the model
#         self.results = self.model.train(
#             data=self.data_yaml,
#             epochs=epochs,
#             imgsz=imgsz,
#             batch=batch_size,
#             device=device,
#             project=project,
#             name=name,
#             patience=10,  # Early stopping patience
#             save=True,
#             save_period=10,
#             cache=True,
#             workers=8,
#             pretrained=True,
#             optimizer='AdamW',
#             lr0=0.001,
#             lrf=0.01,
#             momentum=0.937,
#             weight_decay=0.0005,
#             warmup_epochs=3,
#             warmup_momentum=0.8,
#             warmup_bias_lr=0.1,
#             box=7.5,
#             cls=0.5,
#             dfl=1.5,
#             hsv_h=0.015,
#             hsv_s=0.7,
#             hsv_v=0.4,
#             degrees=0.0,
#             translate=0.1,
#             scale=0.5,
#             shear=0.0,
#             perspective=0.0,
#             flipud=0.0,
#             fliplr=0.5,
#             mosaic=1.0,
#             mixup=0.0,
#             copy_paste=0.0
#         )
        
#         print("Training completed!")
#         return self.results
    
#     def evaluate(self, model_path: str, data_yaml: str = None):
#         """
#         Evaluate the trained model.
        
#         Args:
#             model_path: Path to trained model weights
#             data_yaml: Path to dataset YAML (uses self.data_yaml if None)
#         """
#         model = YOLO(model_path)
#         eval_data = data_yaml or self.data_yaml
        
#         metrics = model.val(data=eval_data)
        
#         print("\n=== Evaluation Results ===")
#         print(f"mAP50-95: {metrics.box.map:.4f}")
#         print(f"mAP50: {metrics.box.map50:.4f}")
#         print(f"mAP75: {metrics.box.map75:.4f}")
        
#         return metrics
    
#     def plot_training_curves(self, save_path: str = 'training_curves.png'):
#         """Plot training metrics."""
#         if self.results is None:
#             print("No training results available. Run train() first.")
#             return
        
#         # Extract metrics
#         metrics = self.results.results
        
#         fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
#         # Plot losses
#         epochs = range(1, len(metrics['train/box_loss']) + 1)
        
#         axes[0, 0].plot(epochs, metrics['train/box_loss'], label='Train Box Loss')
#         axes[0, 0].plot(epochs, metrics['val/box_loss'], label='Val Box Loss')
#         axes[0, 0].set_xlabel('Epoch')
#         axes[0, 0].set_ylabel('Box Loss')
#         axes[0, 0].legend()
#         axes[0, 0].set_title('Box Loss')
        
#         axes[0, 1].plot(epochs, metrics['train/cls_loss'], label='Train Cls Loss')
#         axes[0, 1].plot(epochs, metrics['val/cls_loss'], label='Val Cls Loss')
#         axes[0, 1].set_xlabel('Epoch')
#         axes[0, 1].set_ylabel('Class Loss')
#         axes[0, 1].legend()
#         axes[0, 1].set_title('Class Loss')
        
#         axes[0, 2].plot(epochs, metrics['train/dfl_loss'], label='Train DFL Loss')
#         axes[0, 2].plot(epochs, metrics['val/dfl_loss'], label='Val DFL Loss')
#         axes[0, 2].set_xlabel('Epoch')
#         axes[0, 2].set_ylabel('DFL Loss')
#         axes[0, 2].legend()
#         axes[0, 2].set_title('DFL Loss')
        
#         # Plot metrics
#         axes[1, 0].plot(epochs, metrics['metrics/mAP50-95(B)'], label='mAP50-95')
#         axes[1, 0].plot(epochs, metrics['metrics/mAP50(B)'], label='mAP50')
#         axes[1, 0].set_xlabel('Epoch')
#         axes[1, 0].set_ylabel('mAP')
#         axes[1, 0].legend()
#         axes[1, 0].set_title('mAP Metrics')
        
#         axes[1, 1].plot(epochs, metrics['metrics/precision(B)'], label='Precision')
#         axes[1, 1].plot(epochs, metrics['metrics/recall(B)'], label='Recall')
#         axes[1, 1].set_xlabel('Epoch')
#         axes[1, 1].set_ylabel('Score')
#         axes[1, 1].legend()
#         axes[1, 1].set_title('Precision & Recall')
        
#         # Remove unused subplot
#         axes[1, 2].axis('off')
        
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.show()
#         print(f"Training curves saved to: {save_path}")


# def prepare_dataset_from_yolo_format(dataset_dir: str, output_dir: str = 'prepared_dataset'):
#     """
#     Prepare dataset from YOLO format for training.
    
#     Args:
#         dataset_dir: Directory containing images and labels in YOLO format
#         output_dir: Output directory for prepared dataset
#     """
#     from sklearn.model_selection import train_test_split
#     import shutil
    
#     dataset_path = Path(dataset_dir)
#     output_path = Path(output_dir)
    
#     # Create directories
#     for split in ['train', 'val', 'test']:
#         (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
#         (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
#     # Find all image files
#     image_files = []
#     for ext in ['*.jpg', '*.jpeg', '*.png']:
#         image_files.extend(dataset_path.glob(ext))
#         image_files.extend(dataset_path.glob(ext.upper()))
    
#     # Split dataset
#     train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
#     val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
#     # Copy files
#     for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
#         for img_file in files:
#             # Copy image
#             dest_img = output_path / split / 'images' / img_file.name
#             shutil.copy2(img_file, dest_img)
            
#             # Copy corresponding label file
#             label_file = img_file.with_suffix('.txt')
#             if label_file.exists():
#                 dest_label = output_path / split / 'labels' / label_file.name
#                 shutil.copy2(label_file, dest_label)
    
#     print(f"Dataset prepared at: {output_path}")
#     print(f"Train: {len(train_files)} images")
#     print(f"Val: {len(val_files)} images")
#     print(f"Test: {len(test_files)} images")
    
#     return output_path


# def main():
#     """Main training script."""
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Train Conveyor Belt Damage Detection Model')
#     parser.add_argument('--dataset_dir', type=str, required=True,
#                        help='Directory containing YOLO format dataset')
#     parser.add_argument('--epochs', type=int, default=100,
#                        help='Number of training epochs')
#     parser.add_argument('--batch_size', type=int, default=16,
#                        help='Batch size')
#     parser.add_argument('--imgsz', type=int, default=640,
#                        help='Image size')
#     parser.add_argument('--model', type=str, default='yolov8n.pt',
#                        help='Base YOLO model (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt)')
    
#     args = parser.parse_args()
    
#     # Prepare dataset
#     prepared_dataset = prepare_dataset_from_yolo_format(args.dataset_dir)
    
#     # Create data YAML
#     trainer = ConveyorBeltTrainer(data_yaml_path='dataset.yaml')
#     trainer.create_data_yaml(
#         train_path=str(prepared_dataset / 'train' / 'images'),
#         val_path=str(prepared_dataset / 'val' / 'images'),
#         test_path=str(prepared_dataset / 'test' / 'images'),
#         class_names=['scratch', 'edge_damage'],
#         output_path='dataset.yaml'
#     )
    
#     # Train model
#     trainer.train(
#         epochs=args.epochs,
#         imgsz=args.imgsz,
#         batch_size=args.batch_size,
#         project='runs/train',
#         name='conveyor_belt_detection'
#     )
    
#     # Plot training curves
#     trainer.plot_training_curves()
    
#     # Evaluate best model
#     best_model_path = 'runs/train/conveyor_belt_detection/weights/best.pt'
#     trainer.evaluate(best_model_path)
    
#     # Export to ONNX and TorchScript
#     model = YOLO(best_model_path)
#     model.export(format='onnx', imgsz=args.imgsz)
#     model.export(format='torchscript', imgsz=args.imgsz)
    
#     print("\nTraining pipeline completed!")
#     print(f"Best model saved at: {best_model_path}")


# if __name__ == "__main__":
#     main()