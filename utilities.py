import matplotlib.pyplot as plt
import numpy as np

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
        Trains and validates the specified data model on through the
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
            loss_total = 0
            sample_total = 0
            correct_total = 0

            for inputs, labels in self.training_set:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = model(inputs)
                predictions = outputs > 0.5
                losses = self.loss_criterion(outputs, labels)
                losses.backward()
                self.optimizer.step()
                loss_total += losses.item()
                sample_total += labels.size(0)
                correct_total += (predictions == labels).sum().item()

            training_accuracies[epoch] = correct_total / sample_total
            training_losses[epoch] = loss_total / len(self.training_set)

            # Begin validation loop.
            loss_total = 0
            sample_total = 0
            correct_total = 0
            model.eval()

            for inputs, labels in self.validation_set:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                predictions = outputs > 0.5
                losses = self.loss_criterion(outputs, labels)
                loss_total += losses.item()
                sample_total += labels.size(0)
                correct_total += (predictions == labels).sum().item()

            validation_accuracies[epoch] = correct_total / sample_total
            validation_losses[epoch] = loss_total / len(self.validation_set)

        return TrainingMetrics(training_accuracies,
                               training_losses,
                               validation_accuracies,
                               validation_losses)

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

class PerformanceMetrics:
    '''
    Contains metrics that measure a data model's performance in terms of
    accuracy, precision, recall, and F1 score.
    '''
    def __init__(self,
                 accuracy,
                 precision,
                 recall,
                 f1_score):
        '''
        Initializes a new instance of the PerformanceMetrics class that contains
        the specified accuracy, precision, recall, and F1 score.

        :param self: The instance to initialize.
        :param accuracy: The accuracy of the data model.
        :param precision: The precision of the data model.
        :param recall: The recall of the data model.
        :param f1_score: The F1 score of the data model.
        '''
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
