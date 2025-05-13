import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import ConvNet
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# CUDA/GPU configuration
def setup_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Performance configuration
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        return device
    else:
        print("CUDA not available. Using CPU.")
        return torch.device("cpu")

# Create directory for saving models if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Define transformations for images
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Data augmentation
    transforms.RandomHorizontalFlip(),      # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Determine whether to use GPU and configure the number of workers and batch size
device = setup_cuda()
num_workers = 4 if torch.cuda.is_available() else 2
batch_size = 128 if torch.cuda.is_available() else 64

# Load CIFAR-10 datasets
print("Downloading and preparing CIFAR-10 dataset...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                         shuffle=True, num_workers=num_workers,
                                         pin_memory=torch.cuda.is_available())

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=num_workers,
                                        pin_memory=torch.cuda.is_available())

# CIFAR-10 classes
classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Function to show sample images
def imshow(img):
    img = img / 2 + 0.5  # Denormalize
    npimg = img.numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('results/sample_images.png')

# Function to train the model
def train_model(model, criterion, optimizer, num_epochs=10, device=torch.device("cpu")):
    print(f"Training on: {device}")
    model.to(device)
    
    # Create directory for saving results if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Training metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_acc = 0.0
    total_start_time = time.time()
    
    # Check GPU usage before starting
    if device.type == 'cuda':
        print(f"GPU memory before training: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB / {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB (current/peak)")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training mode
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print statistics every 100 batches
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}, '
                      f'Acc: {100 * correct / total:.2f}%')
                if device.type == 'cuda':
                    print(f"GPU memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB / {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB (current/peak)")
                running_loss = 0.0
        
        # Calculate training accuracy
        train_acc = 100.0 * correct / total
        train_accs.append(train_acc)
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100.0 * correct / total
        val_accs.append(val_acc)
        
        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch + 1} completed in {epoch_time:.2f}s '
              f'- Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%')
    
    total_time = time.time() - total_start_time
    print(f'Training finished in {total_time:.2f}s ({total_time/60:.2f} minutes)')
    print(f'Best validation accuracy: {best_acc:.2f}%')
    
    # Save the final model
    torch.save(model.state_dict(), 'models/cifar10_model.pth')
    
    # Plot metrics
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Training')
    plt.plot(val_accs, label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.savefig('results/training_metrics.png')
    
    return model

# Create the model
model = ConvNet()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
if __name__ == "__main__":
    print("Starting CIFAR-10 image classifier training...")
    
    # Show some sample images
    if not os.path.exists('results'):
        os.makedirs('results')
        
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images[:4]))
    print('Sample classes: ' + ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    # Train model
    trained_model = train_model(model, criterion, optimizer, num_epochs=15, device=device)
    
    # Release CUDA memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory after training: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    print("Training completed. Results have been saved to the 'results/' folder")
    print("The trained model has been saved to 'models/cifar10_model.pth'")
    print("The best model has been saved to 'models/best_model.pth'") 