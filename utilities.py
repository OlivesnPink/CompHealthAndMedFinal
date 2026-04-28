from collections import OrderedDict
from PIL import Image
from sklearn.metrics import multilabel_confusion_matrix
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

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
        self.classes = metadata.columns[1:].to_numpy()

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
            total_loss = 0.0
            total_samples = 0.0
            total_correct = 0.0

            for inputs, labels in self.training_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = model(inputs)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                loss = self.loss_criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_samples += labels.numel()
                total_correct += (predictions == labels).sum().item()

            training_accuracies[epoch] = total_correct / total_samples
            training_losses[epoch] = total_loss / len(self.training_loader)

            # Begin validation loop.
            total_loss = 0.0
            total_samples = 0.0
            total_correct = 0.0
            model.eval()

            for inputs, labels in self.validation_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                loss = self.loss_criterion(outputs, labels)
                total_loss += loss.item()
                total_samples += labels.numel()
                total_correct += (predictions == labels).sum().item()

            validation_accuracies[epoch] = total_correct / total_samples
            validation_losses[epoch] = total_loss / len(self.validation_loader)

        return TrainingMetrics(training_accuracies,
                               training_losses,
                               validation_accuracies,
                               validation_losses)

    def test(self, model):
        '''
        Tests the specified data model through the ModelEvaluator.

        :param self: The ModelEvaluator.
        :param model: The data model to test.
        :return: A numpy array containing confusion matrices for each label.
        '''
        label_count = len(self.testing_loader.dataset.classes)
        matrices = np.zeros((label_count, 2, 2), dtype=int)
        model.eval()

        with torch.no_grad():
            for inputs, labels in self.testing_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                labels = labels.cpu().numpy()
                predictions = predictions.cpu().numpy()
                matrices += multilabel_confusion_matrix(labels, predictions)

        return matrices

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

class dataAugmenter:
    '''
    Helper class to provide data augmentation
    '''
    def __init__(self, image_size: tuple[int, int],
                 norm_mean: tuple[int, int, int],
                 norm_std: tuple[int, int, int],
                 useCutOut: bool = True
                 ):
        '''
        Sets parameters for transforms in properties transform_train and transform_test

        :param self: instance to initialize
        :param norm_mean: means for the red, green, blue channels
        :param norm_std: standard deviations for red, green, blue channels
        :param useCutOut: wether or not cutout is implemented
        '''
        self._transform_train = transforms.compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=60),
            # using list unpacking, random erase is only added to list if useCutOut is true
            *([transforms.RandomErasing(p=0.5,
                                        scale = (0.02, 0.15),
                                        ratio = (0.5, 1.5),
                                        value = 'random')]
                if useCutOut else []),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
        self._transform_test = transforms.compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])

    @property
    def transform_train(self):
        return self._transform_train

    @property
    def transform_test(self):
        return self._transform_test

class _BaseWrapper(object):
    """
    Please modify forward() and backward() according to your task.
    """
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        """
        Simple classification
        """
        self.model.zero_grad()
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        """
        Class-specific backpropagation
        Either way works:
        1. self.logits.backward(gradient=one_hot, retain_graph=True)
        2. (self.logits * one_hot).sum().backward(retain_graph=True)
        """

        one_hot = self._encode_one_hot(ids)
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers  # list

        def forward_hook(key):
            def forward_hook_(module, input, output):
                # Save featuremaps
                self.fmap_pool[key] = output.detach()

            return forward_hook_

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                # Save the gradients correspond to the featuremaps
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook_

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                self.handlers.append(module.register_backward_hook(backward_hook(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def _compute_grad_weights(self, grads):
        return F.adaptive_avg_pool2d(grads, 1)

    def forward(self, image):
        self.image_shape = image.shape[2:]
        return super(GradCAM, self).forward(image)

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = self._compute_grad_weights(grads)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)

        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam
