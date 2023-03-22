from sklearn import svm
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

'''
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=1000,max_depth=2)
clf.fit(train_features,train_label)

y_pred=clf.predict(test_features[:])


from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_label, y_pred))


from sklearn.metrics import classification_report
print(classification_report(test_label, y_pred))

from sklearn.metrics import plot_roc_curve
plot_roc_curve(clf, test_features, test_label) 
plt.show()
'''
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import RocCurveDisplay

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
#svc = SVC(random_state=42)
svc = SVC(kernel=kernel, C=cost, gamma=gamma, coef0=coef0, degree=degree)
svc.fit(train_features, train_label)
svc_disp = RocCurveDisplay.from_estimator(svc, test_features, test_label)

model = XGBClassifier(verbosity = 0, silent=True, n_estimators=100, max_depth=2)
model.fit(train_features, train_label)
ax = plt.gca()
model_disp = RocCurveDisplay.from_estimator(model, test_features, test_label,ax=ax, alpha=0.8)

rfc = RandomForestClassifier(n_estimators=1000, random_state=42,max_depth=2)
rfc.fit(train_features, train_label)
ax = plt.gca()
#rfc_disp = RocCurveDisplay.from_estimator(rfc, test_features, test_label, ax=ax, alpha=0.8)
rfc_disp = RocCurveDisplay.from_estimator(rfc, test_features, test_label, ax=ax, alpha=0.8)
#svc_disp.plot(ax=ax, alpha=0.8)
plt.show()

