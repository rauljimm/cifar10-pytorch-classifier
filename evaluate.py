import torch
import torchvision
import torchvision.transforms as transforms
from model import ConvNet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# GPU/CUDA configuration
def setup_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        
        # Performance configuration
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        return device
    else:
        print("CUDA not available. Using CPU.")
        return torch.device("cpu")

# Get device (GPU or CPU)
device = setup_cuda()

# Create directories if they don't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Define transformations for images
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Configure DataLoader parameters
batch_size = 128 if torch.cuda.is_available() else 64
num_workers = 4 if torch.cuda.is_available() else 2

# Load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=num_workers,
                                        pin_memory=torch.cuda.is_available())

# CIFAR-10 classes
classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def evaluate_model(model_path='models/best_model.pth', device=torch.device("cpu")):
    print(f"Evaluating model: {model_path}")
    
    # Load the model
    model = ConvNet()
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please make sure to train the model first with 'python train.py'")
        return
    
    # Load the model with the correct device
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    print(f"Evaluating on: {device}")
    model.to(device)
    model.eval()
    
    # Monitor GPU usage
    if device.type == 'cuda':
        print(f"GPU memory at evaluation start: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # True labels and predictions
    y_true = []
    y_pred = []
    
    # Evaluate the model
    correct = 0
    total = 0
    start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    
    if device.type == 'cuda':
        start_time.record()
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Save for confusion matrix
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    if device.type == 'cuda':
        end_time.record()
        torch.cuda.synchronize()
        eval_time = start_time.elapsed_time(end_time) / 1000
        print(f"Evaluation time: {eval_time:.2f} seconds")
    
    # Calculate overall accuracy
    accuracy = 100 * correct / total
    print(f'Model accuracy on {total} test images: {accuracy:.2f}%')
    
    # Per-class accuracy
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    print("\nPer-class accuracy:")
    for i in range(10):
        print(f'{classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png')
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    # Save misclassified examples for analysis
    misclassified_analysis(model, device, testloader, classes)
    
    # Free CUDA memory if available
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU memory after evaluation: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    print(f"\nEvaluation completed. Results have been saved to 'results/'")

def misclassified_analysis(model, device, testloader, classes, max_examples=10):
    """Analyze and save examples of misclassified images"""
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Find incorrect predictions
            incorrect_idx = (preds != labels).nonzero(as_tuple=True)[0]
            
            for idx in incorrect_idx:
                if len(misclassified_images) >= max_examples:
                    break
                    
                misclassified_images.append(images[idx].cpu())
                misclassified_labels.append(labels[idx].item())
                misclassified_preds.append(preds[idx].item())
            
            if len(misclassified_images) >= max_examples:
                break
    
    # Display misclassified images
    if len(misclassified_images) > 0:
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(min(max_examples, len(misclassified_images))):
            img = misclassified_images[i] / 2 + 0.5  # Denormalize
            img = img.numpy().transpose((1, 2, 0))
            
            axes[i].imshow(img)
            axes[i].set_title(f'True: {classes[misclassified_labels[i]]}\nPred: {classes[misclassified_preds[i]]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/misclassified_examples.png')
        print(f"Saved {min(max_examples, len(misclassified_images))} examples of misclassified images to 'results/misclassified_examples.png'")

if __name__ == "__main__":
    # Evaluate the best saved model
    evaluate_model('models/best_model.pth', device=device) 