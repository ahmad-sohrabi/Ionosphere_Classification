from CustomFunctions import *


params = {'learning_rate': 0.001, 'activation_fcn': 'relu', 'loss_fcn': 'focal_loss', 'batch_size': 256, 'optimizer': 'adam', 'model_type': 'model2', 'a': 64,
          'b': 64, 'c': 0, 'd': 0, 'e': 0, 'f': 0}
test_loss_KFold, test_acc_KFold, test_precision_KFold, test_recall_KFold, test_fscore_KFold, test_conf_matrix_KFold = train_custom_model(params, True)

print(f"Average Test Loss of All folds is {test_loss_KFold}")
print(f"Average Test Accuracy of All folds is {test_acc_KFold}")
print(f"Average Test Precision of All folds is {test_precision_KFold}")
print(f"Average Test Recall of All folds is {test_recall_KFold}")
print(f"Average Test F-Score of All folds is {test_fscore_KFold}")
print(f"Average Test Confidence matrix of All folds is {test_conf_matrix_KFold}")
plot_confusion_matrix(test_conf_matrix_KFold)
