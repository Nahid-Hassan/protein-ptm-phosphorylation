from sklearn import svm
import numpy as np

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


x_pred = clf.predict(train_features[:])
from sklearn.metrics import confusion_matrix
confusion_matrix(train_label, x_pred)

import matplotlib.pyplot as  plt
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, train_features, train_label)
plt.show()


from sklearn.metrics import accuracy_score
accuracy_score(train_label, x_pred)

from sklearn.metrics import classification_report
print(classification_report(train_label, x_pred))

from sklearn.metrics import plot_roc_curve
plot_roc_curve(clf, train_features, train_label) 
plt.show()