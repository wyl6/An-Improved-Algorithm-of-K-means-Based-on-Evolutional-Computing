import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, recall_score, f1_score
from time import time
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import davies_bouldin_score, silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score
import warnings
warnings.filterwarnings(action="ignore")

##############################################################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Resave dataset
##############################################################################

#def read_data(path):
#    X = np.zeros((210, 8))
#    data = np.array(open(path).readlines())
#    for i in range(210):
#        content = data[i].split('\n')[0].split('\t')
#        while(' ' in content):
#            content.remove(' ')
#        while('' in content):
#            content.remove('')
##        print(content)
#        
#        for j in range(8):
#            X[i][j] = float(content[j])
#        print(X[i])
#    print(X.shape)
#    return X
#
#data_path = '../data/seeds_dataset.txt'
#save_path = '../data/seeds_dataset'
#x = read_data(data_path)
#np.save(save_path, x)

##############################################################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Load dataset
###############################################################################

data = np.load('../data/glass_new.npy') # jaya, very good
X,y = data[:,:9],data[:,9]
y = np.int32(y)

#iris = load_iris()
#X, y = iris['data'], iris['target'] # simulated annealing, very good

##############################################################################
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# KMeans clustering
##############################################################################
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
best_center = None
best_db = 0.0
best_error = 0.0

iters = 100
for i in range(iters):
    kmeans = KMeans(n_clusters=3, y = y)
    kmeans = kmeans.fit(X)
    labels = kmeans.labels_
    prediction = kmeans.get_labels(X)
#    print(y)
#    print(labels)
#    print(prediction)
    C = kmeans.cluster_centers_
#    print(C)
    error = -kmeans.score(X)
    acc = accuracy_score(y, prediction)
#    print('acc', acc)
    avg_acc += acc

#    print('y', y)
#    print('labels', labels)
#    print('predictoin', prediction)
    avg_error += error
    avg_sil += silhouette_score(X, labels, metric='euclidean')
    avg_ri += adjusted_rand_score(y, labels)
    avg_mi += adjusted_mutual_info_score(y, labels)
    avg_hom += homogeneity_score(y, labels)
    db = davies_bouldin_score(X, labels)
    avg_db += db
    if(acc>best_acc):
        best_acc = acc
        best_db = db
        best_center = C
        best_error = error

#print('iters', iters)
print('time', time()-t0)
print('acc', avg_acc/iters)
#print('recall', avg_recall/iters)
#print('f1', avg_f1/iters)
print('error', avg_error/iters)
#print('best_acc', best_acc)
#print('best_center', best_center)
#print('beset_error', best_error)
print('db_index', avg_db/iters) # lower better
print('ri_index', avg_ri/iters) # higher better
print('mi_index', avg_mi/iters) # higher better
print('sil_index', avg_sil/iters) # higher better
#print('hom_index', avg_hom)
#print('best_acc', best_acc)
#print('best_db', best_db)

