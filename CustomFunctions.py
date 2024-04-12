import os

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from NeuralNetClasses import *
import torch.optim as optim
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import datetime
import torch.nn.functional as F
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.weight is not None and self.weight.device != input.device:
            self.weight = self.weight.to(input.device)

        log_pt = F.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = log_pt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)

        if self.weight is not None:
            at = self.weight.gather(0, target)
            log_pt = log_pt * at

        focal_loss = -1 * (1 - pt) ** self.gamma * log_pt

        if self.ignore_index is not None:
            not_ignored = target != self.ignore_index
            focal_loss = focal_loss[not_ignored]

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FBetaLoss(nn.Module):
    def __init__(self, beta=1.0, epsilon=1e-7):
        super(FBetaLoss, self).__init__()
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, inputs, targets):

        probs = F.softmax(inputs, dim=1)

        true_one_hot = torch.nn.functional.one_hot(targets, num_classes=probs.size(1))

        true_positives = torch.sum(true_one_hot * probs, dim=0)
        predicted_positives = torch.sum(probs, dim=0)
        actual_positives = torch.sum(true_one_hot, dim=0)

        precision = true_positives / (predicted_positives + self.epsilon)
        recall = true_positives / (actual_positives + self.epsilon)

        fb = (1 + self.beta ** 2) * precision * recall / ((self.beta ** 2 * precision) + recall + self.epsilon)
        fb_loss = 1 - fb.mean()  # Average over all classes

        return fb_loss


def train(model, optimizer, data_loader, loss_fcn, class_weights_2):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch in data_loader:
        optimizer.zero_grad()
        outputs = model(batch['features'].to(device))

        if loss_fcn == "cross_entropy":
            class_weights = torch.tensor(class_weights_2, dtype=torch.float, device=device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            loss = criterion(outputs, batch['target'].to(device))
        elif loss_fcn == "focal_loss":
            criterion = FocalLoss(gamma=2.0, weight=torch.tensor(class_weights_2), reduction='sum')
            loss = criterion(outputs, batch['target'].to(device))
        else:
            criterion = FBetaLoss(beta=1.0)
            loss = criterion(outputs, batch['target'].to(device))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == batch['target'].to(device)).sum().item()
        total_predictions += batch['target'].size(0)

    epoch_loss = running_loss / len(data_loader)
    epoch_accuracy = correct_predictions / total_predictions
    return epoch_loss, epoch_accuracy


def validate(model, data_loader, loss_fcn, class_weights_2):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch['features'].to(device))

            if loss_fcn == "cross_entropy":
                class_weights = torch.FloatTensor(class_weights_2).to(device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                loss = criterion(outputs, batch['target'].to(device))
            elif loss_fcn == "focal_loss":
                criterion = FocalLoss(gamma=2.0, weight=torch.tensor(class_weights_2), reduction='mean')
                loss = criterion(outputs, batch['target'].to(device))
            else:
                criterion = FBetaLoss(beta=1.0)
                loss = criterion(outputs, batch['target'].to(device))

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.view(-1).cpu().numpy())
            all_targets.extend(batch['target'].view(-1).cpu().numpy())
            correct_predictions += (predicted == batch['target'].to(device)).sum().item()
            total_predictions += batch['target'].size(0)

    epoch_loss = running_loss / len(data_loader)
    epoch_accuracy = correct_predictions / total_predictions
    precision, recall, fscore, _ = precision_recall_fscore_support(all_targets, all_predictions, average='macro',
                                                                   zero_division=0)
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    return epoch_loss, epoch_accuracy, precision, recall, fscore, conf_matrix, model


def define_loss_function_weights(train_index):
    data_frame = pd.read_csv("ionosphere.csv")

    features = data_frame.iloc[:, :-1].values
    targets = data_frame.iloc[:, -1].values

    encoder = LabelEncoder()
    targets = encoder.fit_transform(targets)
    X = features
    y = targets

    train_labels = y[train_index]
    class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(y)])
    weight = 1. / class_sample_count

    return weight


def train_custom_model(params, show_plot):
    dataset = IonosphereDataSet(csv_file='ionosphere.csv')

    X = np.arange(len(dataset))  # Just indices to split
    y = np.array([sample['target'].item() for sample in dataset])

    learning_rate = params['learning_rate']
    activation_fcn = params['activation_fcn']
    model_type = params['model_type']
    loss_fcn = params['loss_fcn']
    batch_size = params['batch_size']
    optimizer_name = params['optimizer']
    a = params["a"]
    b = params["b"]
    c = params["c"]
    d = params["d"]
    e = params["e"]
    f = params["f"]

    NUM_CLASSES = len(np.unique(y))
    NUM_FEATURES = 34
    RANDOM_SEED = 42

    kfold_losses = []
    kfold_accuracy = []
    kfold_precision = []
    kfold_recall = []
    kfold_fscore = []
    kfold_conf_matrix = []
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    start_time = datetime.datetime.now()
    best_accuracy = -np.inf
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        train_subs = Subset(dataset, indices=train_idx)
        val_subs = Subset(dataset, indices=val_idx)

        train_weights = define_loss_function_weights(train_idx)
        val_weights = define_loss_function_weights(val_idx)

        train_loader = DataLoader(train_subs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subs, batch_size=batch_size, shuffle=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_type == "model1":
            model = Multiclass1(NUM_FEATURES, NUM_CLASSES, activation_fcn, a)
            model.to(device)
        elif model_type == "model2":
            model = Multiclass2(NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b)
            model.to(device)
        elif model_type == "model3":
            model = Multiclass3(NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c)
            model.to(device)
        elif model_type == "model4":
            model = Multiclass4(NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d)
            model.to(device)
        elif model_type == "model5":
            model = Multiclass5(NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, e)
            model.to(device)
        elif model_type == "model6":
            model = Multiclass6(NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, e, f)
            model.to(device)


        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
        else:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)


        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)

        epochs = 200

        test_loss_array = np.array([])
        train_loss_array = np.array([])

        test_accuracy_array = np.array([])
        train_accuracy_array = np.array([])

        test_precision_array = np.array([])
        test_recall_array = np.array([])
        test_fscore_array = np.array([])
        test_conf_matrix_array = []

        for epoch in range(epochs):
            train_loss, train_accuracy = train(model, optimizer, train_loader, loss_fcn, train_weights)
            val_loss, val_accuracy, val_precision, val_recall, val_fscore, val_conf_matrix, model_to_analyze = validate(model, val_loader,
                                                                                                      loss_fcn, val_weights)

            if val_accuracy >= best_accuracy:
                # print('Saving Checkpoint ...')
                state = {
                    'net': model_to_analyze.state_dict(),
                    'map': val_accuracy,
                    'epoch': epoch,
                }

                folder = 'checkpoint'
                os.makedirs(folder, exist_ok=True)
                model_name = 'ionosphere'
                path = os.path.join(os.path.abspath(folder), model_name + '.pth')
                torch.save(state, path)
                best_accuracy = val_accuracy
            val_conf_matrix = val_conf_matrix / val_conf_matrix.sum(axis=1, keepdims=True)
            test_loss_array = np.append(test_loss_array, val_loss)
            train_loss_array = np.append(train_loss_array, train_loss)

            test_accuracy_array = np.append(test_accuracy_array, val_accuracy)
            train_accuracy_array = np.append(train_accuracy_array, train_accuracy)

            test_precision_array = np.append(test_precision_array, val_precision)
            test_recall_array = np.append(test_recall_array, val_recall)
            test_fscore_array = np.append(test_fscore_array, val_fscore)
            test_conf_matrix_array.append(val_conf_matrix)

        test_conf_matrix_array = np.array(test_conf_matrix_array)
        kfold_losses.append(np.mean(test_loss_array[-100:]))
        kfold_precision.append(np.mean(test_precision_array[-100:]))
        kfold_recall.append(np.mean(test_recall_array[-100:]))
        kfold_accuracy.append(np.mean(test_accuracy_array[-100:]))
        kfold_fscore.append(np.mean(test_fscore_array[-100:]))
        kfold_conf_matrix.append(np.mean(test_conf_matrix_array[-100:], axis=0))
        if show_plot:
            plt.plot(range(len(train_loss_array)), train_loss_array, label='train loss')
            plt.plot(range(len(test_loss_array)), test_loss_array, label='test loss')
            plt.ylabel("Loss")
            plt.xlabel("Epochs")
            plt.legend()
            plt.show()

            plt.plot(range(len(train_accuracy_array)), train_accuracy_array, label='train accuracy')
            plt.plot(range(len(test_accuracy_array)), test_accuracy_array, label='test accuracy')
            plt.ylabel("Accuracy")
            plt.xlabel("Epochs")
            plt.legend()
            plt.show()
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds()
    print(f"Execution Time is {execution_time} seconds")
    kfold_conf_matrix = np.array(kfold_conf_matrix)
    return np.mean(kfold_losses), np.mean(kfold_accuracy), np.mean(kfold_precision), np.mean(kfold_recall), np.mean(
        kfold_fscore), np.mean(kfold_conf_matrix, axis=0)


class IonosphereDataSet(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)

        self.features = self.data_frame.iloc[:, :-1].values
        self.targets = self.data_frame.iloc[:, -1].values

        self.encoder = LabelEncoder()
        self.targets = self.encoder.fit_transform(self.targets)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'features': self.features[idx], 'target': self.targets[idx]}
        return sample


def plot_confusion_matrix(conf_matrix):
    data_frame = pd.read_csv("ionosphere.csv")

    features = data_frame.iloc[:, :-1].values
    targets = data_frame.iloc[:, -1].values

    encoder = LabelEncoder()
    targets = encoder.fit_transform(targets)
    class_labels = encoder.classes_
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt="g", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')
    plt.show()
