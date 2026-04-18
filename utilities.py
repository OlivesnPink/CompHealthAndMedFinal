from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch

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

    def test(self, model):
        '''
        Tests the specified data model through the ModelEvaluator.

        :param self: The ModelEvaluator.
        :param model: The data model to test.
        :return: A new instance of the PerformanceMetrics class that contains
                 performance metrics.
        '''
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0
        model.eval()

        with torch.no_grad():
            for inputs, labels in self.testing_set:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                predictions = outputs > 0.5
                tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
                total_tp += tp
                total_tn += tn
                total_fp += fp
                total_fn += fn

        return PerformanceMetrics(total_tp, total_tn, total_fp, total_fn)

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

class PerformanceMetrics:
    '''
    Contains metrics that measure a data model's performance.
    '''
    def __init__(self, tp, tn, fp, fn):
        '''
        Initializes a new instance of the PerformanceMetrics class to the
        specified TP, TN, FP, and FN metrics.

        :param self: The instance to initialize.
        :param tp: The true positives.
        :param tn: The true negatives.
        :param fp: The false positives.
        :param fn: The false negatives.
        '''
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def accuracy(self):
        '''
        Returns the accuracy of a data model.

        :param self: The instance containing performance metrics of a data
                     model.
        :return: The accuracy of a data model.
        '''
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def precision(self):
        '''
        Returns the precision of a data model.

        :param self: The instance containing performance metrics of a data
                     model.
        :return: The precision of a data model.
        '''
        return self.tp / (self.tp + self.fp)

    def recall(self):
        '''
        Returns the recall of a data model.

        :param self: The instance containing performance metrics of a data
                     model.
        :return: The recall of a data model.
        '''
        return self.tp / (self.tp + self.fn)

    def specificity(self):
        '''
        Returns the specificity of a data model.

        :param self: The instance containing performance metrics of a data
                     model.
        :return: The specificity of a data model.
        '''
        return self.tn / (self.tn + self.fp)

    def f1_score(self):
        '''
        Returns the F1 score of a data model.

        :param self: The instance containing performance metrics of a data
                     model.
        :return: The F1 score of a data model.
        '''
        precision = self.precision()
        recall = self.recall()
        return 2 * precision * recall / (precision + recall)
