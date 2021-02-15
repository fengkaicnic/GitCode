from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
import numpy as np
import pdb

pdb.set_trace()
y = np.load('poslabel.npy')
probas = np.load('prelabel.npy')

yy = y
pro = probas
tprs = []
aucs = []

fpr = dict()
tpr = dict()
roc_aucc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(yy, pro)
    roc_aucc[i] = auc(fpr[i], tpr[i])
# fpr[0].shape==tpr[0].shape==(21, ), fpr[1].shape==tpr[1].shape==(35, ), fpr[2].shape==tpr[2].shape==(33, ) 
# roc_auc {0: 0.9118165784832452, 1: 0.6029629629629629, 2: 0.7859477124183007}
pdb.set_trace()
plt.figure()
lw = 1
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_aucc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
