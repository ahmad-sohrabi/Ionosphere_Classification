from torch import nn


class Multiclass1(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, activation_fcn, a, dropout_rate=0.5):
        super().__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, a)
        self.batchnorm1 = nn.BatchNorm1d(a)

        if activation_fcn == "relu":
            self.act1 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act1 = nn.Tanh()
        else:
            self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.output = nn.Linear(a, NUM_CLASSES)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.batchnorm1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.output(x)
        return x


class Multiclass2(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, dropout_rate=0.5):
        super().__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, a)
        self.batchnorm1 = nn.BatchNorm1d(a)
        if activation_fcn == "relu":
            self.act1 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act1 = nn.Tanh()
        else:
            self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.hidden2 = nn.Linear(a, b)
        self.batchnorm2 = nn.BatchNorm1d(b)
        if activation_fcn == "relu":
            self.act2 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act2 = nn.Tanh()
        else:
            self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(b, NUM_CLASSES)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.batchnorm1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.hidden2(x)
        x = self.batchnorm2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.output(x)
        return x


class Multiclass3(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, dropout_rate=0.5):
        super().__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, a)
        self.batchnorm1 = nn.BatchNorm1d(a)

        if activation_fcn == "relu":
            self.act1 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act1 = nn.Tanh()
        else:
            self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.hidden2 = nn.Linear(a, b)
        self.batchnorm2 = nn.BatchNorm1d(b)

        if activation_fcn == "relu":
            self.act2 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act2 = nn.Tanh()
        else:
            self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.hidden3 = nn.Linear(b, c)
        self.batchnorm3 = nn.BatchNorm1d(c)

        if activation_fcn == "relu":
            self.act3 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act3 = nn.Tanh()
        else:
            self.act3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.output = nn.Linear(c, NUM_CLASSES)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.batchnorm1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.hidden2(x)
        x = self.batchnorm2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.hidden3(x)
        x = self.batchnorm3(x)
        x = self.act3(x)
        x = self.dropout3(x)

        x = self.output(x)
        return x


class Multiclass4(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, dropout_rate=0.5):
        super().__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, a)
        self.batchnorm1 = nn.BatchNorm1d(a)

        if activation_fcn == "relu":
            self.act1 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act1 = nn.Tanh()
        else:
            self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.hidden2 = nn.Linear(a, b)
        self.batchnorm2 = nn.BatchNorm1d(b)
        if activation_fcn == "relu":
            self.act2 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act2 = nn.Tanh()
        else:
            self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.hidden3 = nn.Linear(b, c)
        self.batchnorm3 = nn.BatchNorm1d(c)
        if activation_fcn == "relu":
            self.act3 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act3 = nn.Tanh()
        else:
            self.act3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.hidden4 = nn.Linear(c, d)
        self.batchnorm4 = nn.BatchNorm1d(d)
        if activation_fcn == "relu":
            self.act4 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act4 = nn.Tanh()
        else:
            self.act4 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout(dropout_rate)

        self.output = nn.Linear(d, NUM_CLASSES)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.batchnorm1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.hidden2(x)
        x = self.batchnorm2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.hidden3(x)
        x = self.batchnorm3(x)
        x = self.act3(x)
        x = self.dropout3(x)

        x = self.hidden4(x)
        x = self.batchnorm4(x)
        x = self.act4(x)
        x = self.dropout4(x)

        x = self.output(x)
        return x


class Multiclass5(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, e, dropout_rate=0.5):
        super().__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, a)
        self.batchnorm1 = nn.BatchNorm1d(a)
        if activation_fcn == "relu":
            self.act1 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act1 = nn.Tanh()
        else:
            self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.hidden2 = nn.Linear(a, b)
        self.batchnorm2 = nn.BatchNorm1d(b)
        if activation_fcn == "relu":
            self.act2 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act2 = nn.Tanh()
        else:
            self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.hidden3 = nn.Linear(b, c)
        self.batchnorm3 = nn.BatchNorm1d(c)
        if activation_fcn == "relu":
            self.act3 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act3 = nn.Tanh()
        else:
            self.act3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.hidden4 = nn.Linear(c, d)
        self.batchnorm4 = nn.BatchNorm1d(d)
        if activation_fcn == "relu":
            self.act4 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act4 = nn.Tanh()
        else:
            self.act4 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout(dropout_rate)

        self.hidden5 = nn.Linear(d, e)
        self.batchnorm5 = nn.BatchNorm1d(e)
        if activation_fcn == "relu":
            self.act5 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act5 = nn.Tanh()
        else:
            self.act5 = nn.LeakyReLU()
        self.dropout5 = nn.Dropout(dropout_rate)

        self.output = nn.Linear(e, NUM_CLASSES)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.batchnorm1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.hidden2(x)
        x = self.batchnorm2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.hidden3(x)
        x = self.batchnorm3(x)
        x = self.act3(x)
        x = self.dropout3(x)

        x = self.hidden4(x)
        x = self.batchnorm4(x)
        x = self.act4(x)
        x = self.dropout4(x)

        x = self.hidden5(x)
        x = self.batchnorm5(x)
        x = self.act5(x)
        x = self.dropout5(x)

        x = self.output(x)
        return x


class Multiclass6(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, activation_fcn, a, b, c, d, e, f, dropout_rate=0.5):
        super().__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, a)
        self.batchnorm1 = nn.BatchNorm1d(a)
        if activation_fcn == "relu":
            self.act1 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act1 = nn.Tanh()
        else:
            self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.hidden2 = nn.Linear(a, b)
        self.batchnorm2 = nn.BatchNorm1d(b)
        if activation_fcn == "relu":
            self.act2 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act2 = nn.Tanh()
        else:
            self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.hidden3 = nn.Linear(b, c)
        self.batchnorm3 = nn.BatchNorm1d(c)
        if activation_fcn == "relu":
            self.act3 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act3 = nn.Tanh()
        else:
            self.act3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.hidden4 = nn.Linear(c, d)
        self.batchnorm4 = nn.BatchNorm1d(d)
        if activation_fcn == "relu":
            self.act4 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act4 = nn.Tanh()
        else:
            self.act4 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout(dropout_rate)

        self.hidden5 = nn.Linear(d, e)
        self.batchnorm5 = nn.BatchNorm1d(e)
        if activation_fcn == "relu":
            self.act5 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act5 = nn.Tanh()
        else:
            self.act5 = nn.LeakyReLU()
        self.dropout5 = nn.Dropout(dropout_rate)

        self.hidden6 = nn.Linear(e, f)
        self.batchnorm6 = nn.BatchNorm1d(f)
        if activation_fcn == "relu":
            self.act6 = nn.ReLU()
        elif activation_fcn == "tanh":
            self.act6 = nn.Tanh()
        else:
            self.act6 = nn.LeakyReLU()
        self.dropout6 = nn.Dropout(dropout_rate)

        self.output = nn.Linear(f, NUM_CLASSES)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.batchnorm1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.hidden2(x)
        x = self.batchnorm2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.hidden3(x)
        x = self.batchnorm3(x)
        x = self.act3(x)
        x = self.dropout3(x)

        x = self.hidden4(x)
        x = self.batchnorm4(x)
        x = self.act4(x)
        x = self.dropout4(x)

        x = self.hidden5(x)
        x = self.batchnorm5(x)
        x = self.act5(x)
        x = self.dropout5(x)

        x = self.hidden6(x)
        x = self.batchnorm6(x)
        x = self.act6(x)
        x = self.dropout6(x)

        x = self.output(x)
        return x



