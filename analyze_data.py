from sklearn.preprocessing import LabelEncoder
import pandas as pd

data_frame = pd.read_csv("ionosphere.csv")

features = data_frame.iloc[:, :-1].values
targets = data_frame.iloc[:, -1].values

encoder = LabelEncoder()
targets = encoder.fit_transform(targets)
print(f"Class Labels are: {encoder.classes_}")

X = features
y = targets

number_of_ones = sum(y)
number_of_zeros = len(y) - number_of_ones

print(f"Number of Bad Labels in dataset is {number_of_zeros}")
print(f"Number of Good Labels in dataset is {number_of_ones}")

