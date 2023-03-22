import numpy as np
from ML import SVM_Classifier
from pubscripts import read_code_ml, save_file, draw_plot, calculate_prediction_metrics

train = "Encoding_result/train_ptm.txt" #Training dataset
indep = "Encoding_result/test_ptm.txt" #Independent dataset


format = "csv" # choices=['tsv', 'svm', 'csv', 'weka']
kernel = "rbf" #choices=['linear', 'poly', 'rbf', 'sigmoid']
auto = 'False' #auto optimize parameters
batch = 0.5 #random select part (batch * samples) samples for parameters optimization
degree = 3 #set degree in polynomial kernel function (default 3)
gamma = 'auto' #set gamma in polynomial/rbf/sigmoid kernel function (default 1/k)
coef0 = 0 #set coef0 in polynomial/rbf/sigmoid kernel function (default 0)
cost = 1  #set the parameter cost value (default 1)
fold = 5 #n-fold cross validation mode (default 5-fold cross-validation, 1 means jack-knife cross-validation)
out = 'SVM' #set prefix for output score file

if gamma == None:
    gamma = 'auto'
X, y, independent = 0, 0, np.array([])
X, y = read_code_ml.read_code(train, format='%s' % format)

if indep:
    ind_X, ind_y = read_code_ml.read_code(indep, format='%s' % format)
    independent = np.zeros((ind_X.shape[0], ind_X.shape[1] + 1))
    independent[:, 0], independent[:, 1:] = ind_y, ind_X

para_info, cv_res, ind_res = SVM_Classifier.SVM_Classifier(X, y, indep=independent, fold=fold, batch=batch,
                                                       auto=auto, kernel=kernel, degree=degree,
                                                       gamma=gamma, coef0=coef0, C=cost)
classes = sorted(list(set(y)))

if len(classes) == 2:
    save_file.save_CV_result_binary(cv_res, '%s_abir_CV.txt' % out, para_info)
    mean_auc = draw_plot.plot_roc_cv(cv_res, '%s_abir_ROC_CV.png' % out, label_column=0, score_column=2)
    mean_auprc = draw_plot.plot_prc_CV(cv_res, '%s_abir_PRC_CV.png' % out, label_column=0, score_column=2)
    cv_metrics = calculate_prediction_metrics.calculate_metrics_cv(cv_res, label_column=0, score_column=2,)
    save_file.save_prediction_metrics_cv(cv_metrics, '%s_abir_metrics_CV.txt' % out)

    if indep:
        save_file.save_IND_result_binary(ind_res, '%s_abir_IND.txt' % out, para_info)
        ind_auc = draw_plot.plot_roc_ind(ind_res, '%s_abir_ROC_IND.png' % out, label_column=0, score_column=2)
        ind_auprc = draw_plot.plot_prc_ind(ind_res, '%s_abir_PRC_IND.png' % out, label_column=0, score_column=2)
        ind_metrics = calculate_prediction_metrics.calculate_metrics(ind_res[:, 0], ind_res[:, 2])
        save_file.save_prediction_metrics_ind(ind_metrics, '%s_abir_metrics_IND.txt' %out)

if len(classes) > 2:
    save_file.save_CV_result(cv_res, classes, '%s_abir_CV.txt' % out, para_info)
    cv_metrics = calculate_prediction_metrics.calculate_metrics_cv_muti(cv_res, classes, label_column=0)
    save_file.save_prediction_metrics_cv_muti(cv_metrics, classes, '%s_abir_metrics_CV.txt' % out)

    if indep:
        save_file.save_IND_result(ind_res, classes, '%s_abir_IND.txt' % out, para_info)
        ind_metrics = calculate_prediction_metrics.calculate_metrics_ind_muti(ind_res, classes, label_column=0)
        save_file.save_prediction_metrics_ind_muti(ind_metrics, classes, '%s_abir_metrics_IND.txt' % out)




