import matplotlib.pyplot as plt
import torch

def get_accuracy(predicted, labels):
    '''
    Returns a tuple containing the number of elements within the specified
    tensor of ground truth labels and the number of elements within the
    specified tensor of predicted labels that match the ground truth labels.

    :param predicted: The tensor containing ground truth labels.
    :param labels: The tensor containing predicted labels.
    :returns: A tuple containing the number of elements within the specified
              tensor of ground truth labels and the number of elements within
              the specified tensor of predicted labels that match ground truth
              labels.
    '''
    batch_length = labels.size(0)
    correct_count = (predicted == labels).sum().item()
    return batch_length, correct_count

class ModelTrainer:
    '''
    Provides methods that train and validate data models according to a set of
    configurations.
    '''
    def __init__(self,
                 training_data_loader,
                 validation_data_loader,
                 loss_criterion,
                 optimizer,
                 device):
        '''
        Initializes a new instance of the ModelTrainer class that trains data
        models on data loaded from the specified data loaders into the specified
        device using specified loss criterion and optimizer.

        :param self: The instance to initialize.
        :param training_data_loader: The data loader containing the training
                                     set.
        :param validation_data_loader: The data loader containing the validation
                                       set.
        :param loss_criterion: The loss function to use during training.
        :param optimizer: The optimizer to use during training.
        :param device: The device to load data into.
        '''
        self.training_data_loader = training_data_loader
        self.validation_data_loader = validation_data_loader
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.device = device

    def validate(self, model):
        '''
        Validates the specified data model on the validation set contained by
        the ModelTrainer.

        :param self: The ModelTrainer.
        :param model: The data model to validate.
        :return: A tuple containing validation accuracy and loss metrics.
        '''
        loss = 0
        sample_total = 0
        correct_total = 0
        model.eval()

        for inputs, labels in self.validation_data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = model(inputs)
            # TODO: Change to multilabel classification.
            _, predicted = torch.max(outputs, 1)
            loss += self.loss_criterion(outputs, labels).item()
            batch_length, correct_count = get_accuracy(predicted, labels)
            sample_total += batch_length
            correct_total += correct_count

        accuracy = correct_total / sample_total
        loss /= len(self.validation_data_loader)
        return accuracy, loss

    def train(self, model, epoch_count):
        '''
        Trains and validates the specified data model on through the
        ModelTrainer.

        :param self: The ModelTrainer.
        :param model: The data model to train and validate.
        :return: A new instance of the TrainingResults class that contains
                 training accuracy and loss metrics.
        '''
        training_accuracies = [0] * epoch_count
        training_losses = [0] * epoch_count
        validation_accuracies = [0] * epoch_count
        validation_losses = [0] * epoch_count

        for epoch in range(epoch_count):
            training_loss = 0
            sample_total = 0
            correct_total = 0

            for inputs, labels in self.training_data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = model(inputs)
                loss_tensor = self.loss_criterion(outputs, labels)
                loss_tensor.backward()
                self.optimizer.step()
                # TODO: Change to multilabel classification.
                _, predicted = torch.max(outputs, 1)
                batch_length, correct_count = get_accuracy(predicted, labels)
                training_loss += loss_tensor.item()
                sample_total += batch_length
                correct_total += correct_count

            training_accuracies[epoch] = correct_total / sample_total
            training_losses[epoch] = training_loss / len(self.training_data_loader)
            validation_accuracies[epoch], validation_losses[epoch] = self.validate(model)

        return TrainingResults(training_accuracies,
                               training_losses,
                               validation_accuracies,
                               validation_losses)

class TrainingResults:
    '''
    Contains metrics that measure accuracy and loss across training epochs.
    '''
    def __init__(self,
                 training_accuracies,
                 training_losses,
                 validation_accuracies,
                 validation_losses):
        '''
        Initializes a new instance of the TrainingResults class that contains
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

    def plot_accuracies(self):
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

    def plot_losses(self):
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
