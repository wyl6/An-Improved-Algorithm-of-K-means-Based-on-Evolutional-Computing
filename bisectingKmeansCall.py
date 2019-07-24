from numpy import mat
import numpy as np
import bisectingKmeans as kMeans
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import davies_bouldin_score, silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score
from time import time
import warnings
warnings.filterwarnings(action="ignore")
#==================================================================
'''10.1 datasets'''
#dataMat = mat(kMeans.loadDataSet('testSet2.txt'))

iris = load_iris()
dataMat, y = iris['data'], iris['target'] # simulated annealing, very good

#==================================================================
'''10.2 二分k均值算法'''


t0 = time()
avg_acc = 0.0
#avg_recall = 0.0
#avg_f1 = 0.0
avg_error = 0.0
avg_db = 0.0
avg_sil = 0.0
avg_ri = 0.0
avg_mi = 0.0
avg_hom = 0.0
best_acc = 0.0
worst_acc = 1.0
best_center = None
best_db = 0.0
best_error = 0.0
iters = 100

for j in range(iters):
    centoids,clusterAssment = kMeans.biKmeans(dataMat.copy(),3,calDis=kMeans.calDistance)
#    print(clusterAssment)
    labels = clusterAssment[:,0].tolist()
    labels = np.array([x[0] for x in labels])
#    print(labels)
#    print(j)
    
    predictions = np.zeros(dataMat.shape[0])
    for i in range(3):
        predictions[labels == i] = np.argmax(np.bincount(y[labels == i]))
##    print(predictions)
    acc = accuracy_score(y, predictions)
    avg_acc += acc
    error = np.sum(clusterAssment[:, 1])
    avg_error += error
    avg_sil += silhouette_score(dataMat, predictions, metric='euclidean')
    avg_ri += adjusted_rand_score(y, predictions)
    avg_mi += adjusted_mutual_info_score(y, predictions)
    avg_hom += homogeneity_score(y, predictions)
    avg_db += davies_bouldin_score(dataMat, predictions)
    
    if(acc>best_acc):
        best_acc = acc
    if(acc<worst_acc):
        worst_acc = acc
        
#    print(j, 'acc', acc)
print('time', time()-t0)
print('acc', avg_acc/iters)
#print('best_acc', best_acc)
#print('worst_acc', worst_acc)
print('error', avg_error/iters)
print('db_index', avg_db/iters) # lower better
print('ri_index', avg_ri/iters) # higher better
print('mi_index', avg_mi/iters) # higher better
print('sil_index', avg_sil/iters) # higher better
#==================================================================
'''10.3 对地图上的点进行聚类'''
#kMeans.clusterClubs(5)
'''
time 12.093095541000366
acc 0.8653333333333336
error 85.86812143029081
db_index 0.6677426946549463
ri_index 0.6728579731471188
mi_index 0.6765116690538839
sil_index 0.538448040846951
'''




