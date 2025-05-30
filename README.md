# pytorch_modular
Collection of functions to use with PyTorch

# PyTorch Helper Functions

This repository holds a collection of helper functions designed to streamline the process of training, evaluating, and visualizing results with PyTorch models.

## Functions Included

Here's a description of the functions currently available in this repository:

### `create_dataloaders`

Creates training and testing `DataLoader` instances from specified directories.

- `train_dir`: Path to the training data directory.
- `test_dir`: Path to the testing data directory.
- `transform`: torchvision transforms to apply to images.
- `batch_size`: Number of samples per batch.
- `num_workers`: Number of workers for the DataLoader.

### `train_step`

Performs a single training step for a PyTorch model.

- `model`: The PyTorch model to train.
- `dataloader`: The DataLoader for the training data.
- `loss_fn`: The loss function to use.
- `optimizer`: The optimizer to use.
- `device`: The device to train on (`"cuda"` or `"cpu"`).

### `test_step`

Performs a single testing step for a PyTorch model.

- `model`: The PyTorch model to test.
- `dataloader`: The DataLoader for the testing data.
- `loss_fn`: The loss function to use.
- `device`: The device to test on (`"cuda"` or `"cpu"`).

### `train`

Trains and tests a PyTorch model over a specified number of epochs.

- `model`: The PyTorch model to train and test.
- `train_dataloader`: The DataLoader for the training data.
- `test_dataloader`: The DataLoader for the testing data.
- `optimizer`: The optimizer to use.
- `loss_fn`: The loss function to use.
- `epochs`: The number of epochs to train for.
- `device`: The device to train and test on (`"cuda"` or `"cpu"`).

### `pred_and_plot_image`

Makes a prediction on a target image using a trained model and plots the image with the prediction.

- `model`: The trained PyTorch model.
- `class_names`: A list of class names.
- `image_path`: The path to the image to predict on.
- `image_size`: The size to resize the image to (default: (224, 224)).
- `transform`: Optional torchvision transform to apply to the image.
- `device`: The device to make the prediction on.

### `plot_predictions`

Plots training and testing data and optionally predictions.

- `train_data`: Training data.
- `train_labels`: Training labels.
- `test_data`: Testing data.
- `test_labels`: Testing labels.
- `predictions`: Optional predictions to plot.

### `plot_decision_boundary`

Plots the decision boundary of a model.

- `model`: The PyTorch model to plot the decision boundary for.
- `X`: Input features.
- `y`: True labels.

### `plot_loss_curves`

Plots the training and testing loss and accuracy curves from training results.

- `results`: A dictionary containing training and testing loss and accuracy.

### `explore_dir`

Explores a directory and prints the number of subdirectories and files.

- `dir_path`: The path to the directory to explore.

### `calculate_accuracy`

Calculates the accuracy between true and predicted labels.

- `y_true`: True labels.
- `y_pred`: Predicted labels.

### `print_train_time`

Prints the training time.

- `start`: Start time of training.
- `end`: End time of training.
- `device`: Device used for training.

### `set_seeds`

Sets random seeds for reproducibility in PyTorch.

- `seed`: The seed value (default: 42).

### `download_data`

Downloads a zip file from a URL and unzips it.

- `source`: The URL of the zip file.
- `destination`: The directory to unzip the data to.
- `remove_source`: Whether to remove the downloaded zip file (default: True).

## Future Additions

More useful functions for PyTorch development will be added to this repository in the future. Stay tuned!
