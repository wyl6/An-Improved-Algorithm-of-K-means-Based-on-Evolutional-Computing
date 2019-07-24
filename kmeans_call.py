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
#
#data = np.load('../data/glass_new.npy') # jaya, very good
#X,y = data[:,:9],data[:,9]
#y = np.int32(y)

iris = load_iris()
X, y = iris['data'], iris['target'] # simulated annealing, very good

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
#    print(type(y), type(prediction))
    C = kmeans.cluster_centers_
#    print('C', C)
    error = -kmeans.score(X)
    acc = accuracy_score(y, prediction)
#    print('acc', acc)
    avg_acc += acc

#    print('y', y)
#    print('labels', labels)
#    print('predictoin', prediction)
    avg_error += error
    avg_sil += silhouette_score(X, prediction, metric='euclidean')
    avg_ri += adjusted_rand_score(y, prediction)
    avg_mi += adjusted_mutual_info_score(y, prediction)
    avg_hom += homogeneity_score(y, prediction)
    db = davies_bouldin_score(X, prediction)
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

'''
kmeans++
time 1.123659610748291
acc 0.8832000000000003
error 80.79761855092566
db_index 0.6564794004938124
ri_index 0.7174068435573223
mi_index 0.7336306340288832
sil_index 0.5557758156445205

time 1.1151678562164307
acc 0.8828666666666675
error 80.79782977091718
db_index 0.6566827521269175
ri_index 0.716712035577341
mi_index 0.7328679180391363
sil_index 0.5556944452576794

time 1.1571311950683594
acc 0.8852000000000005
error 80.13177502353186
db_index 0.6592188688935734
ri_index 0.7187541616019649
mi_index 0.7353105771142556
sil_index 0.5544284477664942

time 1.1466662883758545
acc 0.8809333333333336
error 81.4096584690157
db_index 0.6541073123423782
ri_index 0.7153298234882406
mi_index 0.7310962663866316
sil_index 0.5569817796639341

time 1.123753309249878
acc 0.8825333333333342
error 80.79804099090872
db_index 0.6568861037600224
ri_index 0.71601722759736
mi_index 0.7321052020493889
sil_index 0.5556130748708386

time 1.1793007850646973
acc 0.8831333333333338
error 80.7706745022755
db_index 0.6566224202913549
ri_index 0.7171809541410991
mi_index 0.733355964948393
sil_index 0.5557213877925825

kmeans++ KSJ
time 6.718405485153198
acc 0.8891333333333341
error 78.85410279803956
db_index 0.6645337770778734
ri_index 0.721483691735708
mi_index 0.7387621718521323
sil_index 0.5517937454822135

kmeans++ KSAJ
time 53.22230863571167
acc 0.8893333333333342
error 78.8539760660446
db_index 0.6644117660980107
ri_index 0.7219005765236965
mi_index 0.7392198014459808
sil_index 0.5518425677143179

kmeans++ KFJ
time 6.180590629577637
acc 0.8874000000000011
error 79.4927910567916
db_index 0.6617339768425502
ri_index 0.7206052922548232
mi_index 0.7375702756760174
sil_index 0.5531680558951425

kmeans++ KFAJ
time 204.97144484519958
acc 0.8929999999999988
error 78.85165264613761
db_index 0.662174898133857
ri_index 0.7295434643034896
mi_index 0.7476096773332008
sil_index 0.5527376419695679

'''



























