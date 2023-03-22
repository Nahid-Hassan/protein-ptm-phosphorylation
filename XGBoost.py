
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as  plt

train_path = "./Encoding_result/train_ptm.txt"
#train_path = "/content/drive/MyDrive/PTM_Project/iLearn_AbirModifiedFinal/Encoding_result/train_ptm.txt"
test_path = "./Encoding_result/test_ptm.txt"
#test_path = "/content/drive/MyDrive/PTM_Project/iLearn_AbirModifiedFinal/Encoding_result/test_ptm.txt"

# read and convert train path data to numpy array
train_data = np.loadtxt(train_path, delimiter=',')
#print (train_data[:])
#print(train_data.shape)

# divided train_data two featues and label first index is label
train_label = train_data[:, 0]
train_features = train_data[:, 1:]

test_data = np.loadtxt(test_path, delimiter=',')
#print (test_data[2:4])

# divided test_data two featues and label first index is label
test_label = test_data[:, 0]
test_features = test_data[:, 1:]
# fit model no training data

model = XGBClassifier(verbosity = 0, silent=True, n_estimators=100, max_depth=6)
model.fit(train_features, train_label)

# make predictions for test data
y_pred = model.predict(test_features)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(test_label, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.metrics import plot_roc_curve
plot_roc_curve(model, test_features, test_label) 
plt.show()