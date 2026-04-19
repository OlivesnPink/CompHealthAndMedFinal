from PIL import Image
from sklearn.metrics import confusion_matrix
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
        labels = torch.tensor(row[1:].to_numpy())
        return image, labels

class ModelEvaluator:
    '''
    Provides methods that evaluate data models according to a set of
    configurations.
    '''
    def __init__(self,
                 training_set,
                 validation_set,
                 testing_set,
                 loss_criterion,
                 optimizer,
                 device):
        '''
        Initializes a new instance of the ModelEvaluator class that evaluates
        data models on data loaded from the specified data loaders into the
        specified device using specified loss criterion and optimizer.

        :param self: The instance to initialize.
        :param training_set: The data loader containing the training set.
        :param validation_set: The data loader containing the validation set.
        :param testing_set: The data loader containing the testing set.
        :param loss_criterion: The loss function to use during training.
        :param optimizer: The optimizer to use during training.
        :param device: The device to load data into.
        '''
        self.training_set = training_set
        self.validation_set = validation_set
        self.testing_set = testing_set
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, model, epoch_count):
        '''
        Trains and validates the specified data model through the
        ModelEvaluator.

        :param self: The ModelEvaluator.
        :param model: The data model to train and validate.
        :return: A new instance of the TrainingMetrics class that contains
                 training accuracy and loss metrics.
        '''
        training_accuracies = np.zeros(epoch_count)
        training_losses = np.zeros(epoch_count)
        validation_accuracies = np.zeros(epoch_count)
        validation_losses = np.zeros(epoch_count)

        for epoch in range(epoch_count):
            # Begin training loop.
            total_loss = 0
            total_samples = 0
            total_correct = 0

            for inputs, labels in self.training_set:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = model(inputs)
                predictions = outputs > 0.5
                loss = self.loss_criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_samples += labels.size(0)
                total_correct += (predictions == labels).sum().item()

            training_accuracies[epoch] = total_correct / total_samples
            training_losses[epoch] = total_loss / len(self.training_set)

            # Begin validation loop.
            total_loss = 0
            total_samples = 0
            total_correct = 0
            model.eval()

            for inputs, labels in self.validation_set:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                predictions = outputs > 0.5
                loss = self.loss_criterion(outputs, labels)
                total_loss += loss.item()
                total_samples += labels.size(0)
                total_correct += (predictions == labels).sum().item()

            validation_accuracies[epoch] = total_correct / total_samples
            validation_losses[epoch] = total_loss / len(self.validation_set)

        return TrainingMetrics(training_accuracies,
                               training_losses,
                               validation_accuracies,
                               validation_losses)

    def test(self, model):
        '''
        Tests the specified data model through the ModelEvaluator.

        :param self: The ModelEvaluator.
        :param model: The data model to test.
        :return: A confusion matrix containing performance metrics.
        '''
        matrix = np.zeros((46, 46), dtype=int)
        model.eval()

        with torch.no_grad():
            for inputs, labels in self.testing_set:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                predictions = outputs > 0.5
                matrix += confusion_matrix(labels, predictions)

        return matrix

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
