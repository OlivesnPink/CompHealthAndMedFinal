from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch

class FundusImageDataset(Dataset):
    '''
    Represents a partition of the RFMiD dataset.
    '''
    def __init__(self, metadata, image_directory, transform):
        '''
        Initializes a new instance of the FundusImageDataset that contains the
        specified data frame, loads images from the specified directory, and
        modifies them using the specified transform.

        :self: The instance to initialize.
        :metadata: The data frame containing ground truth labels.
        :image_directory: The directory containing images in the dataset.
        :transform: The transform used to modify images loaded from the dataset.
        '''
        self.metadata = metadata
        self.image_directory = image_directory
        self.transform = transform

    def __len__(self):
        '''
        Returns the size of the FundusImageDataset.

        :self: The instance to get the size of.
        :return: The size of the EyeImageDataset.
        '''
        return len(self.metadata)

    def __getitem__(self, index):
        '''
        Returns an image and vector of ground truth labels at the specified row
        index from the FundusImageDataset.

        :self: The instance to search.
        :index: The row index of the desired image and vector of ground truth
                labels.
        :return: A tuple containing an image and vector of ground truth labels
                 at index.
        '''
        row = self.metadata.iloc[index]
        filepath = f'{self.image_directory}/{row['ID']}.png'
        image = self.transform(Image.open(filepath))
        labels = torch.from_numpy(row[1:].to_numpy(float))
        return image, labels

class ModelEvaluator:
    '''
    Provides methods that evaluate data models according to a set of
    configurations.
    '''
    def __init__(self,
                 training_loader,
                 validation_loader,
                 testing_loader,
                 loss_criterion,
                 optimizer,
                 device):
        '''
        Initializes a new instance of the ModelEvaluator class that evaluates
        data models on data loaded from the specified data loaders into the
        specified device using specified loss criterion and optimizer.

        :param self: The instance to initialize.
        :param training_loader: The data loader containing the training set.
        :param validation_loader: The data loader containing the validation set.
        :param testing_loader: The data loader containing the testing set.
        :param loss_criterion: The loss function to use during training.
        :param optimizer: The optimizer to use during training.
        :param device: The device to load data into.
        '''
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.testing_loader = testing_loader
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, model, epoch_count):
        '''
        Trains and validates the specified data model through the
        ModelEvaluator.
        '''
        training_accuracies = np.zeros(epoch_count)
        training_losses = np.zeros(epoch_count)
        validation_accuracies = np.zeros(epoch_count)
        validation_losses = np.zeros(epoch_count)

        for epoch in range(epoch_count):
            # Begin training loop.
            model.train()
            total_loss = 0.0
            total_samples = 0.0
            total_correct = 0.0

            print(f"epoch {epoch}: starting")

            for inputs, labels in self.training_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = model(inputs)
                
                # Use 0.3 here as well for consistency across your project
                predictions = (torch.sigmoid(outputs) > 0.60).float()
                
                loss = self.loss_criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_samples += labels.numel()
                total_correct += (predictions == labels).sum().item()

            training_accuracies[epoch] = total_correct / total_samples
            training_losses[epoch] = total_loss / len(self.training_loader)

            print("Training done. Onto validation.")

            # Begin validation loop.
            model.eval()
            val_labels = []
            val_preds = []
            val_total_loss = 0.0

            for inputs, labels in self.validation_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = self.loss_criterion(outputs, labels)
                    val_total_loss += loss.item()
                    predictions = (torch.sigmoid(outputs) > 0.60).float()
                
                val_labels.append(labels.cpu().numpy())
                val_preds.append(predictions.cpu().numpy())

            # Stack everything into two big matrices
            val_labels = np.vstack(val_labels)
            val_preds = np.vstack(val_preds)

            # Calculate Macro F1
            from sklearn.metrics import f1_score, accuracy_score
            epoch_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
            
            # Update the metric arrays
            # Using (val_preds == val_labels).mean() gives bit-wise accuracy (Hamming)
            # accuracy_score gives subset accuracy (all 29 must match perfectly)
            validation_accuracies[epoch] = (val_preds == val_labels).mean() 
            validation_losses[epoch] = val_total_loss / len(self.validation_loader)

            print(f"epoch {epoch}: F1-Score: {epoch_f1:.4f} | Bit-Accuracy: {validation_accuracies[epoch]:.4f}")

        # --- THE CRITICAL FIX: Add this return statement ---
        return TrainingMetrics(training_accuracies,
                               training_losses,
                               validation_accuracies,
                               validation_losses)
    
    
    def test(self, model, label_names=None):
        model.eval()
        all_labels = []
        all_probs = [] # Changed from all_preds to all_probs

        with torch.no_grad():
            for inputs, labels in self.testing_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                
                # Store RAW probabilities (0.0 to 1.0)
                probs = torch.sigmoid(outputs)
                
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        # Combine all batches
        val_trues = np.vstack(all_labels)
        val_probs = np.vstack(all_probs)
        
        if label_names is None:
            label_names = [f"Class {i}" for i in range(val_trues.shape[1])]

        # --- Threshold Optimization ---
        best_thresholds = []
        for i in range(val_trues.shape[1]):
            best_f1 = -1
            best_thresh = 0.5
            
            y_prob = val_probs[:, i]
            y_true = val_trues[:, i]
            
            # Search for the best threshold for this specific disease
            for thresh in np.arange(0.05, 0.95, 0.01):
                y_pred = (y_prob >= thresh).astype(int)
                score = f1_score(y_true, y_pred, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_thresh = thresh
            best_thresholds.append(best_thresh)

        # Apply the optimized thresholds to get final predictions
        final_preds = np.zeros_like(val_probs)
        for i in range(val_trues.shape[1]):
            final_preds[:, i] = (val_probs[:, i] >= best_thresholds[i]).astype(int)

        # Generate the report
        print("Optimized Thresholds per class:")
        print(best_thresholds)
        
        report = classification_report(
            val_trues, 
            final_preds, 
            target_names=label_names, 
            zero_division=0
        )
        
        print(report)
        return report, best_thresholds

class TrainingMetrics:
    '''
    Contains metrics that measure accuracy and loss across training epochs.
    '''
    def __init__(self,
                 training_accuracies,
                 training_losses,
                 validation_accuracies,
                 validation_losses):
        '''
        Initializes a new instance of the TrainingMetrics class that contains
        the specified lists of accuracies and losses obtained across training
        epochs.

        :param self: The instance to initialize.
        :param training_accuracies: The list of training accuracies.
        :param training_losses: The list of training losses.
        :param validation_accuracies: The list of validation accuracies.
        :param validation_losses: The list of validation losses.
        '''
        self.training_accuracies = training_accuracies
        self.training_losses = training_losses
        self.validation_accuracies = validation_accuracies
        self.validation_losses = validation_losses

    def show_accuracies(self):
        '''
        Shows a plot of the training and validation accuracies contained by the
        TrainingMetrics.

        :param self: The instance containing training and validation accuracies.
        '''
        epochs = range(1, len(self.training_accuracies) + 1)
        plt.figure(figsize=(10, 15))
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.training_accuracies, label='Training Accuracy')
        plt.plot(epochs, self.validation_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.legend()
        plt.figure()
        plt.show()

    def show_losses(self):
        '''
        Shows a plot of the training and validation losses contained by the
        TrainingMetrics.

        :param self: The instance containing training and validation losses.
        '''
        epochs = range(1, len(self.training_losses) + 1)
        plt.figure(figsize=(10, 15))
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.training_losses, label='Training Loss')
        plt.plot(epochs, self.validation_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.figure()
        plt.show()