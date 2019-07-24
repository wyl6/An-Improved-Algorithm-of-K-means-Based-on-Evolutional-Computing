# coding = utf-8
from numpy import *
#==================================================================
'''
10.1 K均值聚类算法
numpy.random.rand(m,n):对于给定形状的数组a(m,n)，将其填充在一个均匀分布的样本[0,1)中
numpy.random.rand可能和random冲突，最好这样写：
import numpy as np
np.numpy.rand(m,n)
'''
#==================================================================
def loadDataSet(filename):
    '''
    加载数据
    '''
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        floLine = list(map(float,curLine))
        dataMat.append(floLine)
    return dataMat

def calDistance(vecA,vecB):
    '''
    计算点到质心的欧式距离
    '''
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet,k):
    '''
    随机获得k个聚类中心
    '''
    n = shape(dataSet)[1]
    
    randCtr = mat(zeros((k,n)))
    for i in range(n):
        minI = min(dataSet[:,i])
        maxI = max(dataSet[:,i])
        rangI = float(maxI - minI)
        randCtr[:,i] = mat(minI + rangI * random.rand(k,1))
    return randCtr

def kMeans(dataSet,k,calDis = calDistance,createCtr = randCent):
#    print('dataSet',dataSet[0])
    m = shape(dataSet)[0]
    centeroids = createCtr(dataSet,k)    # k个聚类中心
    clusterAssment = mat(zeros((m,2)))  # 簇分配结果矩阵
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf;minIndex = -1
            for j in range(k):
                disIJ = calDis(dataSet[i,:],centeroids[j,:])
                if(disIJ < minDist):
                    minDist = disIJ;minIndex = j
            if(clusterAssment[i,0] != minIndex):clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        # print(centeroids)
        # 更新质心位置
        for center in range(k):
            centerSet = dataSet[nonzero(clusterAssment[:,0].A == center)[0]]
            centeroids[center,:] = mean(centerSet,axis=0)   # 按照列方向求平均值到行
    return centeroids,clusterAssment


#==================================================================
# 10.2 二分k均值算法
'''
将所有点看成一个簇
当簇数＜k
    对于每一个簇
        计算总误差
        对簇进行而均值聚类
        计算聚类后的总误差
    选择使得误差最小的那个簇进行划分操作
'''
#==================================================================

def biKmeans(dataSet,k,calDis = calDistance):
    '''
    二分聚类
    '''
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centeroid0 = mean(dataSet,axis = 0).tolist()[0]
    centerList = [centeroid0]   # 初始聚类
#    print(centerList)
    for j in range(m):
        clusterAssment[j,1] = calDis(mat(centeroid0),dataSet[j,:]) ** 2
#    print(clusterAssment)
    while(len(centerList) < k):   # 当质心数小于k
        lowestSSE = inf
        for i in range(len(centerList)):  # 找到划分的最优簇进行二分聚类
            # 聚类后误差
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0],:]
#            print('i ', i, ' shape ', shape(ptsInCurrCluster))
            centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,calDis)
            sseSplit = sum(splitClustAss[:,1])
            # 聚类前误差
            sseNoSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            # 更新此次聚类最值
            if(sseSplit+sseNoSplit < lowestSSE):
                bestCentToSplit = i
                bestClustAss = splitClustAss.copy()
                bestNewCents = centroidMat #　二分聚类得到的质心矩阵
                lowestSSE = sseSplit+sseNoSplit
        # 将二分聚类的结果一类沿用父亲标号，一类沿用总数标号
        '''
        质心为1的赋值一定要在质心为0的前面，否则为质心为0的赋值以后，其质心为1，然后又重新赋值，数据聚类会出现错误
        具体错误就是当bestCentToSplit长度恰好为1时，所有质心为1的会被总数len(centerList)标记，这样质心中就没有1了
        下次for i in range(len(centerList))中，质心1的簇就是空集，会出现错误ValueError: min() arg is an empty sequence
        '''
        
        if(nonzero(bestClustAss[:, 0].A == 1)[0].shape[0] == 0): continue
        if(nonzero(bestClustAss[:, 0].A == 0)[0].shape[0] == 0): continue
    
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centerList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        # print('The best split center to split is: ',bestCtrToSplit)
        # print('The len of bestClassAssment is: ',len(bestClassAssment))
        # 用得到的两个聚类中心，更新原来的质心，添加新的质心
        centerList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centerList.append(bestNewCents[1,:].tolist()[0])
        # 用第i个簇分为j,k两个簇的数据将原来的数据替换掉，包括对应簇中心和误差
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return mat(centerList),clusterAssment
