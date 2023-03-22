from sklearn import svm
import numpy as np
train_path = "./Encoding_result/train_ptmForPAAC.txt"
#train_path = "./Encoding_result/train_ptm.txt"
#train_path = "/content/drive/MyDrive/PTM_Project/iLearn_AbirModifiedFinal/Encoding_result/train_ptm.txt"
#test_path = "./Encoding_result/test_ptm.txt"
#test_path = "/content/drive/MyDrive/PTM_Project/iLearn_AbirModifiedFinal/Encoding_result/test_ptm.txt"
test_path = "./Encoding_result/test_ptmForPAAC.txt"
# read and convert train path data to numpy array
train_data = np.loadtxt(train_path, delimiter=',')
#print (train_data[:])
#print(train_data.shape)

# divided train_data two featues and label first index is label
train_label = train_data[:, 0]
train_features = train_data[:, 1:]

#print (train_label[2:4])
#print (train_features[2:4])


format = "csv" # choices=['tsv', 'svm', 'csv', 'weka']
kernel = "rbf" #choices=['linear', 'poly', 'rbf', 'sigmoid']
auto = 'False' #auto optimize parameters
batch = 0.4 #random select part (batch * samples) samples for parameters optimization
degree = 3 #set degree in polynomial kernel function (default 3)
gamma = 'auto' #set gamma in polynomial/rbf/sigmoid kernel function (default 1/k)
coef0 = 0 #set coef0 in polynomial/rbf/sigmoid kernel function (default 0)
cost = 1  #set the parameter cost value (default 1)
fold = 8 #n-fold cross validation mode (default 5-fold cross-validation, 1 means jack-knife cross-validation)
out = 'SVM' #set prefix for output score file


clf = svm.SVC(kernel=kernel, C=cost, gamma=gamma, coef0=coef0, degree=degree)
clf.fit(train_features, train_label)

# read and convert train path data to numpy array
test_data = np.loadtxt(test_path, delimiter=',')
#print (test_data[2:4])

# divided test_data two featues and label first index is label
test_label = test_data[:, 0]
test_features = test_data[:, 1:]



y_pred = clf.predict(test_features[:])
from sklearn.metrics import confusion_matrix

confusion_matrix(test_label, y_pred)

import matplotlib.pyplot as  plt
from sklearn.metrics import plot_confusion_matrix
#plot_confusion_matrix(clf, test_features, test_label)
#plt.show()

from sklearn.metrics import accuracy_score
print(accuracy_score(test_label, y_pred))

from sklearn.metrics import classification_report
print(classification_report(test_label, y_pred))

from sklearn.metrics import plot_roc_curve
plot_roc_curve(clf, test_features, test_label) 
plt.show()
