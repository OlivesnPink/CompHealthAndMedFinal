import torch

def get_accuracy(predicted, labels):
    """
    Returns a tuple containing the number of elements within the specified
    tensor of ground truth labels and the number of elements within the
    specified tensor of predicted labels that match the ground truth labels.

    :param predicted: The tensor containing ground truth labels.
    :param labels: The tensor containing predicted labels.
    :returns: A tuple containing the number of elements within the specified
              tensor of ground truth labels and the number of elements within
              the specified tensor of predicted labels that match ground truth
              labels.
    """
    batch_length = labels.size(0)
    correct_count = (predicted == labels).sum().item()
    return batch_length, correct_count

def evaluate(model, device, data_loader, loss_criterion):
    """
    Evaluates the specified model on data loaded from the specified data loader
    into the specified device using specified loss criterion.

    :param model: The data model to evaluate.
    :param device: The device to load data into.
    :param data_loader: The data loader.
    :param loss_criterion: The loss function to use during evaluation.
    :return: A tuple containing the loss and accuracy metrics as a result of
             evaluation.
    """
    loss = 0
    sample_total = 0
    correct_total = 0
    model.eval()
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        # TODO: Change to multilabel classification.
        _, predicted = torch.max(outputs, 1)
        loss += loss_criterion(outputs, labels).item()
        batch_length, correct_count = get_accuracy(predicted, labels)
        sample_total += batch_length
        correct_total += correct_count
    accuracy = correct_total / sample_total
    loss = loss / len(data_loader)
    return loss, accuracy
