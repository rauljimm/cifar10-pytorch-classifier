import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import ConvNet
import argparse
import os
import numpy as np
import time

# GPU/CUDA configuration
def setup_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Performance configuration
        torch.backends.cudnn.benchmark = True
        
        return device
    else:
        print("CUDA not available. Using CPU.")
        return torch.device("cpu")

# Get device (GPU or CPU)
device = setup_cuda()

# CIFAR-10 classes
classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

def predict_image(image_path, model_path='models/best_model.pth', device=torch.device("cpu")):
    """
    Predicts the class of an image using the trained model.
    
    Args:
        image_path: Path to the image to classify
        model_path: Path to the trained model
        device: Device to run inference on (CPU/GPU)
    
    Returns:
        class_name: Name of the predicted class
        probability: Probability of the prediction
    """
    # Check if results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please make sure to train the model first with 'python train.py'")
        return None, 0
        
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, 0
    
    # Load the model
    model = ConvNet()
    
    # Load model weights with the correct device
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    # Show device information
    print(f"Running prediction on: {device}")
    if device.type == 'cuda':
        print(f"GPU memory before prediction: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Load and transform the image
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(device)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, 0
    
    # Perform prediction
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        probability = torch.nn.functional.softmax(output, dim=1)
    inference_time = time.time() - start_time
    
    # Get class and probability
    class_idx = predicted.item()
    class_name = classes[class_idx]
    prob = probability[0][class_idx].item() * 100
    
    print(f"Inference time: {inference_time*1000:.2f} ms")
    
    # Visualize the image and prediction
    plt.figure(figsize=(8, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(np.array(image))
    plt.axis('off')
    plt.title(f"Image: {os.path.basename(image_path)}")
    
    plt.subplot(2, 1, 2)
    # Show probabilities for all classes
    probs = probability.squeeze().cpu().numpy() * 100
    plt.barh(classes, probs)
    plt.xlabel('Probability (%)')
    plt.xlim(0, 100)
    plt.title(f'Prediction: {class_name} ({prob:.2f}%)')
    
    plt.tight_layout()
    plt.savefig('results/prediction_result.png')
    
    print(f"Prediction: {class_name} with {prob:.2f}% confidence")
    print(f"Result saved to 'results/prediction_result.png'")
    
    # Free GPU memory if applicable
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return class_name, prob

def bulk_predict(directory_path, model_path='models/best_model.pth', limit=10, device=torch.device("cpu")):
    """
    Performs predictions for all images in a directory.
    
    Args:
        directory_path: Path to directory with images
        model_path: Path to the trained model
        limit: Maximum number of images to process
        device: Device to run inference on (CPU/GPU)
    """
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Get all images in the directory
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                  if os.path.isfile(os.path.join(directory_path, f)) and 
                  any(f.lower().endswith(ext) for ext in extensions)]
    
    if not image_files:
        print(f"No images found in {directory_path}")
        return
    
    # Limit the number of images to process
    image_files = image_files[:limit]
    
    print(f"Processing {len(image_files)} images...")
    print(f"Device used: {device}")
    
    # Create a figure to show all predictions
    n_cols = 3
    n_rows = (len(image_files) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Load the model once
    model = ConvNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    # Show CUDA information
    if device.type == 'cuda':
        print(f"GPU memory at start of processing: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Measure total time
    total_start_time = time.time()
    
    # Process each image
    for i, image_path in enumerate(image_files):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        try:
            # Load and transform the image
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)
            input_tensor = input_tensor.to(device)
            
            # Perform prediction
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                probability = torch.nn.functional.softmax(output, dim=1)
            
            # Get class and probability
            class_idx = predicted.item()
            class_name = classes[class_idx]
            prob = probability[0][class_idx].item() * 100
            
            # Show image and prediction
            ax.imshow(np.array(image))
            ax.set_title(f"{class_name} ({prob:.1f}%)")
            ax.axis('off')
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            ax.text(0.5, 0.5, f"Error: {str(e)[:50]}...", ha='center', va='center')
            ax.axis('off')
    
    # Hide empty axes
    for i in range(len(image_files), n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/bulk_predictions.png')
    
    total_time = time.time() - total_start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/len(image_files)*1000:.2f} ms")
    
    # Free CUDA memory if available
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU memory after processing: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    print(f"Predictions completed. Result saved to 'results/bulk_predictions.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR-10 Image Classifier')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Command for predicting a single image
    predict_parser = subparsers.add_parser('image', help='Predict a single image')
    predict_parser.add_argument('--image', type=str, required=True, help='Path to the image to classify')
    predict_parser.add_argument('--model', type=str, default='models/best_model.pth', help='Path to the trained model')
    
    # Command for predicting a batch of images
    bulk_parser = subparsers.add_parser('directory', help='Predict multiple images from a directory')
    bulk_parser.add_argument('--directory', type=str, required=True, help='Path to the directory with images')
    bulk_parser.add_argument('--model', type=str, default='models/best_model.pth', help='Path to the trained model')
    bulk_parser.add_argument('--limit', type=int, default=10, help='Maximum number of images to process')
    
    # Option to use CPU even if GPU is available
    parser.add_argument('--cpu', action='store_true', help='Force CPU use even if GPU is available')
    
    args = parser.parse_args()
    
    # Determine the device to use (respect --cpu option if provided)
    active_device = torch.device("cpu") if args.cpu else device
    
    # Execute the appropriate command
    if args.command == 'image':
        predict_image(args.image, args.model, device=active_device)
    elif args.command == 'directory':
        bulk_predict(args.directory, args.model, args.limit, device=active_device)
    else:
        parser.print_help() 