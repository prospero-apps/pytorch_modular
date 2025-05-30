"""
Utility functions for PyTorch model training, plotting, and saving.
"""
import torch
import torchvision
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import zipfile
import requests
from typing import List, Tuple
from PIL import Image

def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image and plots the image with the prediction.

    Args:
        model: A PyTorch model to make a prediction on.
        class_names: A list of class names to map the prediction to.
        image_path: The path to the target image.
        image_size: The size to resize the image to (default: (224, 224)).
        transform: A torchvision transform to apply to the image (default: None).
        device: The device to make the prediction on (default: "cuda" if available, else "cpu").
    """
    # Open the image
    img = Image.open(image_path)

    # Create a transform if none is provided
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # Put the model on the target device
    model.to(device)

    # Turn on inference mode
    model.eval()
    with torch.inference_mode():
        # Transform the image and add a batch dimension
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Make a prediction
        target_image_pred = model(transformed_image.to(device))

    # Calculate the prediction probabilities
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Get the predicted class label
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot the image and the prediction
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False)

def plot_predictions(
    train_data: torch.Tensor, 
    train_labels: torch.Tensor, 
    test_data: torch.Tensor, 
    test_labels: torch.Tensor, 
    predictions: torch.Tensor = None
):
    """
    Plots training and testing data and optionally predictions.

    Args:
        train_data: Training data.
        train_labels: Training labels.
        test_data: Testing data.
        test_labels: Testing labels.
        predictions: Optional predictions to plot.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot testing data
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    # Plot predictions if provided
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})

def plot_decision_boundary(model: torch.nn.Module, 
                           X: torch.Tensor, 
                           y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)

    Args:
        model: PyTorch model to plot the decision boundary for.
        X: Input features.
        y: True labels.
    """
    # Move model and data to CPU
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Set up prediction boundaries and make a grid of 101x101 points
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features to get predictions on
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Change predictions to labels (required for plotting)
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # multiclass
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def plot_loss_curves(results: Dict[str, List]):
    """Plots training and testing loss and accuracy curves.

    Args:
        results: A dictionary containing "train_loss", "train_acc", "test_loss", and "test_acc" lists.
    """
    # Get the loss and accuracy values
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # Get the number of epochs
    epochs = range(len(results["train_loss"]))

    # Plot loss
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    
def explore_dir(dir_path: str):
    """
    Explores a directory and prints the number of directories and files within it.

    Args:
        dir_path: The path to the directory to explore.
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def calculate_accuracy(y_true: torch.Tensor, 
                       y_pred: torch.Tensor):
    """
    Calculates accuracy between true labels and predicted labels.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        The accuracy as a percentage.
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start: float, 
                     end: float, 
                     device: torch.device = None):
    """Prints the training time.

    Args:
        start: Start time of training.
        end: End time of training.
        device: Device used for training (e.g., "cuda" or "cpu").
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model's state_dict to a target directory.

    Args:
        model: A PyTorch model to save.
        target_dir: The directory to save the model to.
        model_name: The name of the model file to save. Should end with ".pt" or ".pth".
    """
    # Create target directory if it doesn't exist
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)
    
def set_seeds(seed: int = 42):
    """Sets random seeds for PyTorch and CUDA.

    Args:
        seed: The seed value (default: 42).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads data from a source URL and unzips it to a destination directory.

    Args:
        source: The URL of the data to download.
        destination: The directory to unzip the data to.
        remove_source: Whether to remove the downloaded zip file (default: True).

    Returns:
        The path to the unzipped data directory.
    """
    data_path = Path("data/")
    image_path = data_path / destination

    # Check if data directory exists
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download the data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip the data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove the source zip file if requested
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path
