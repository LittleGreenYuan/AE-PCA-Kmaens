import os
import csv
import matplotlib.pyplot as plt
import math
from sklearn import datasets
import pandas as pd
from numpy import nan as NaN
from scipy import stats
from scipy import fftpack
import random
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import spectral_clustering
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle  ##python自带的迭代器模块
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.random as rnd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
# 这个Axes3D要输入来，否则下面画3D散点图时参数projection = '3d'会报错
from mpl_toolkits.mplot3d import Axes3D
 
 
def Normalization(arr):
    arr=arr.reshape(arr.size,1)
    x = (arr - np.min(arr))/(np.max(arr)- np.min(arr))
    #x = (arr - np.average(arr)) / np.std(arr);
    return x


#欧氏距离计算
def distEclud(x,y):
    return np.sqrt(np.sum((x-y)**2))
# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet,k):
    # 获取样本数与特征值
    m,n = dataSet.shape#把数据集的行数和列数赋值给m,n
    # 初始化质心,创建(k,n)个以零填充的矩阵
    centroids = np.zeros((k,n))
    # 循环遍历特征值
    for i in range(k):
        index = int(np.random.uniform(0,m))
        # 计算每一列的质心,并将值赋给centroids
        centroids[i,:] = dataSet[index,:]
        # 返回质心
    return centroids


# k均值聚类
def KMeans(dataSet,k):
    m = np.shape(dataSet)[0]
    # 初始化一个矩阵来存储每个点的簇分配结果
    # clusterAssment包含两个列:一列记录簇索引值,第二列存储误差(误差是指当前点到簇质心的距离,后面会使用该误差来评价聚类的效果)
    clusterAssment = np.mat(np.zeros((m,2)))
    clusterChange = True

    # 创建质心,随机K个质心
    centroids = randCent(dataSet,k)
    # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
    while clusterChange:
        clusterChange = False

        #遍历所有样本（行数）
        for i in range(m):
            minDist = 100000.0
            #minDist = 1000.0
            minIndex = -1
            # 遍历所有数据找到距离每个点最近的质心,
            # 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
            for j in range(k):
                # 计算数据点到质心的距离
                # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
                distance = distEclud(centroids[j,:],dataSet[i,:])
                # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)
                clusterAssment[i,:] = minIndex,minDist
        # 遍历所有质心并更新它们的取值
        for j in range(k):
            # 通过数据过滤来获得给定簇的所有点
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]
            # 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
            centroids[j,:] = np.mean(pointsInCluster,axis=0)
    print("Congratulation,cluster complete!")
    # 返回所有的类质心与点分配结果
    return centroids,clusterAssment

def showCluster(dataSet,k,centroids,clusterAssment):
    m,n = dataSet.shape
    #if n != 2:
       # print("数据不是二维的")
        #return 1

    mark = ['or','ob','og','ok','^r','+r','sr','dr','<r','pr','1r','1b','1g','xr','hr','hb','hg','xb','4r','4b','4g','xg','*r','*b']
    if k > len(mark):
        print("k值太大了")
        return 1
    #绘制所有样本
    for i in range(m):
        markIndex = int(clusterAssment[i,0])
        plt.plot(dataSet[i,0],dataSet[i,1],mark[markIndex])

    #mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    mark = ['*r','*b','*g','*k','*r','+r','sr','dr','<r','pr','1r','1b','1g','xr','hr','hb','hg','xb','4r','4b','4g','xg','or','ob']
    #绘制质心
    for i in range(k):
        plt.plot(centroids[i,0],centroids[i,1],mark[i])

    plt.show()



def loadDataSet(fileName):
    #主数据库内部的文件搜寻
#主数据库路径
    filePath_main =fileName
#数据库中有多少文件夹
    fileNames_main=os.listdir(filePath_main)
#生成这些文件夹的对应路径
    fileDir_main = [os.path.join(filePath_main, fileName) for fileName in fileNames_main]
#确定文件夹的数量
    K=len(fileNames_main)
    j=0
    dataSet=np.zeros(2048)
    full_dataSet=np.zeros(2078)
    featuresSet=np.zeros(30)
    part_features=np.zeros(12)
    step=2;#每隔1个取一个点
    fft_list=list(range(0,1024,step))
    fft_list=np.float_(fft_list)
    fft_list=fft_list.reshape(fft_list.size,1)
    datare_fft=np.zeros(len(fft_list))
    dataim_fft=np.zeros(len(fft_list))
    
    
    dataLable=[0]
    for i in range(K):
        fileNames = os.listdir(fileDir_main[i])
#生成这些文件夹的对应路径
        fileDir = [os.path.join(fileDir_main[i], fileName) for fileName in fileNames]
#确定文件夹的数量
        K_k=len(fileNames)
        for ii in range(K_k):
            fileNames_in = os.listdir(fileDir[ii])
#生成这些文件夹的对应路径
            fileDir_in = [os.path.join(fileDir[ii], fileName) for fileName in fileNames_in]
#确定文件夹的数量
            K_in=len(fileNames_in)
            for iii in range(K_in):
                df_train = pd.read_csv(fileDir_in[iii],header=None)#读取csv文件
                x=np.float_(df_train)#获得数据
                #实部特征
                re_mean = []#均值
                re_std = []#标准差
                re_var = []#方差
                re_min = []#最小值
                re_max = []#最大值
                re_median = []#中值
                re_skew = []#偏度
                re_kuri = []#峰度
                re_iqr = []#四分位间距
                re_df = []#均方根
                re_rms = []#波形因子
                re_ff = []
                re_par = []#峰度因子
                re_pulse = []#脉冲因子
                re_ppv =[]#峰峰值
                re_fft = []#fft
                re_energy =[]#能量
                refft=np.zeros(len(fft_list))
                #虚部特征
                im_mean = []#均值
                im_std = []#标准差
                im_var = []#方差
                im_min = []#最小值
                im_max = []#最大值
                im_median = []#中值
                im_skew = []#偏度
                im_kuri = []#峰度
                im_iqr = []
                im_df=[]#均方根
                im_rms = []
                im_ff = []
                im_par = []#峰度因子
                im_pulse = []#脉冲因子
                im_ppv =[]#峰峰值
                im_fft = []#fft
                im_energy =[]#能量
                imfft=np.zeros(len(fft_list))
                
                re_x=Normalization(x[:,0])#获得实部信息
                re_x=re_x.reshape(re_x.size,1)
                re_mean=np.mean(re_x)
                re_std=np.std(re_x)
                re_var=np.var(re_x)
                re_min=np.min(re_x)
                re_max=np.max(re_x)
                re_median=np.median(re_x)
                re_skew=stats.skew(re_x)
                re_kuri=stats.kurtosis(re_x)
                
                lower_q=np.quantile(re_x,0.25,interpolation='lower')#四分位间距
                higher_q=np.quantile(re_x,0.75,interpolation='higher')
                re_iqr=higher_q - lower_q
                
                re_df=np.mean(re_x)
                re_rms=math.sqrt(pow(re_df,2) + pow(re_std,2))
                re_ff=re_rms / (abs(re_x).mean())
                re_par=(max(re_x)) / re_rms
                re_pulse=(max(re_x)) / (abs(re_x).mean())
                re_ppv=np.max(re_x)-np.min(re_x)
                #re_energy = fftpack.fft(re_x)#能量
                re_fft = np.abs(fftpack.fft(re_x))
                
                
                im_x=Normalization(x[:,1])#获得虚部信息
                im_x=im_x.reshape(im_x.size,1)
                im_mean=np.mean(im_x)
                im_std=np.std(im_x)
                im_var=np.var(im_x)
                im_min=np.min(im_x)
                im_max=np.max(im_x)
                im_median=np.median(im_x)
                im_skew=stats.skew(im_x)
                im_kuri=stats.kurtosis(im_x)
                
                lower_q=np.quantile(im_x,0.25,interpolation='lower')#四分位间距
                higher_q=np.quantile(im_x,0.75,interpolation='higher')
                im_iqr=higher_q - lower_q
                
                im_df=np.mean(im_x)
                im_rms=math.sqrt(pow(im_df,2) + pow(im_std,2))
                im_ff=im_rms / (abs(im_x).mean())
                im_par=(max(im_x)) / im_rms
                im_pulse=(max(im_x)) / (abs(im_x).mean())
                im_ppv=np.max(im_x)-np.min(im_x)
                #im_energy = fftpack.fft(im_x)#能量
                im_fft = np.abs(fftpack.fft(im_x))
                #
                features=[]
                features=[re_mean,re_std,re_var,re_min,re_max,re_median,re_skew,re_kuri,re_iqr,re_df,re_rms,re_ff,re_par,re_pulse,re_ppv,im_mean,im_std,im_var,im_min,im_max,im_median,im_skew,im_kuri,im_iqr,im_df,im_rms,im_ff,im_par,im_pulse,im_ppv]
                
                features=np.float_(features)
                features=features.reshape(features.size)
                
                new_features=[re_mean,re_kuri,re_iqr,re_df,re_rms,re_pulse,im_mean,im_kuri,im_iqr,im_df,im_rms,im_pulse]
                new_features=np.float_(new_features)
                new_features=new_features.reshape(new_features.size)
                #re_x=Normalization(re_x)
                #im_x=Normalization(im_x)
                x=np.hstack((re_x,im_x))
                x=x.reshape(x.size)
                
                full_x=x
                full_x = np.append(full_x,features)
                full_x=full_x.reshape(full_x.size)
                     
                     
                     
                re_fft=re_fft.T
                im_fft=im_fft.T
                refft=[0]
                imfft=[0]
                for q in range(0,len(re_fft[0]),step):
                    refft.append(re_fft[0,q])
                for q in range(0,len(im_fft[0]),step):
                    imfft.append(im_fft[0,q])
                                
                refft=np.float_(refft)
                refft =np.delete(refft,0,axis=0)
                refft=refft.reshape(1,refft.size)
                #refft=Normalization(refft)
                
                imfft=np.float_(imfft)
                imfft =np.delete(imfft,0,axis=0)
                imfft=imfft.reshape(1,imfft.size)
                #imfft=Normalization(imfft)
                
                datare_fft=np.vstack((datare_fft,refft.reshape(refft.size)))
                dataim_fft=np.vstack((dataim_fft,imfft.reshape(imfft.size)))
                
                
                
                dataSet=np.vstack((dataSet,x))
                full_dataSet=np.vstack((full_dataSet,full_x))
                featuresSet=np.vstack((featuresSet,features))
                part_features=np.vstack((part_features,new_features))
                dataLable.append(int(j))
        j=j+1
    dataSet =np.delete(dataSet,0,axis=0)
    full_dataSet =np.delete(full_dataSet,0,axis=0)
    featuresSet =np.delete(featuresSet,0,axis=0)
    datare_fft =np.delete(datare_fft,0,axis=0)
    dataim_fft =np.delete(dataim_fft,0,axis=0)
    part_features=np.delete(part_features,0,axis=0)
    dataLable=np.int_(dataLable)
    dataLable =np.delete(dataLable,0,axis=0)
    data_fft= np.hstack((part_features,datare_fft))
    data_fft= np.hstack((data_fft,dataim_fft))
    #data_fft= np.hstack((part_features,datare_fft))
    #data_fft= np.hstack((dataim_fft,datare_fft))
    #data_fft= np.hstack((data_fft,x))
    data_fft= np.hstack((data_fft,dataSet))
    return dataSet,full_dataSet,featuresSet,dataLable,dataim_fft,datare_fft,data_fft,part_features

dataSet,full_dataSet,featuresSet,dataLable,dataim_fft,datare_fft,data_fft,part_features = loadDataSet("..\\data") 

rnd.seed(4)
m = 200
w1, w2 = 0.1, 0.3
noise = 0.1

'''
angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
data = np.empty((m, 3))
data[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * rnd.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)
'''

#data=np.hstack((full_dataSet,data_fft))
data=full_dataSet
# 进行训练、测试数据的预处理，这里主要是为了使输入网络的数据具有一致性
scaler = StandardScaler()
X_train = scaler.fit_transform(data[:2900])
X_test = scaler.transform(data[2400:])
'''
X_train = scaler.fit_transform(data[:6000])
X_test = scaler.transform(data[2100:]) 
'''

n_inputs = len(data[0])
n_hidden = 20
# 输出的维度跟输入的维度一直是自编码网络的一个特点
n_outputs = n_inputs
learning_rate = 0.01
X = tf.placeholder(tf.float32, shape=(None, n_inputs))
hidden = tf.layers.dense(X, n_hidden)
outputs = tf.layers.dense(hidden, n_outputs)
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(reconstruction_loss)
 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
 
n_iteration = 1000
codings = hidden
 
for i in range(n_iteration):
    sess.run(training_op, feed_dict={X: X_train})
coding_tensroVal,outputsVal = sess.run([codings,outputs], feed_dict={X: X_test})
codings_val = coding_tensroVal

X=codings_val

pca = PCA(n_components=2)   #降到2维
pca.fit(codings_val)                 #训练
X=pca.fit_transform(codings_val)   #降维后的数据



k = 5
centroids,clusterAssment = KMeans(X,k)
showCluster(X,k,centroids,clusterAssment)
A=np.int_(clusterAssment[:,0])

#k=n_clusters_
cluster=clusterAssment[:,0]
cluster=cluster.reshape(k,1200)
counter = np.zeros(shape=(k,k))
for ci in range(len(cluster)):
    for cj in range(k):
        indices_zero = np.nonzero(cluster[ci,:] == cj)
        T=np.int_(indices_zero[1])
        counter[ci,cj]=len(T)        
right_number=0
for ri in range(len(counter)):
    tp = np.argmax(counter[ri,:])
    right_number=counter[ri,tp]+right_number

right=right_number/len(X)

''' 
fig = plt.figure()
plt.plot(codings_val[:, 0], codings_val[:, 1], "b.")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
fig1 = plt.figure()
ax = plt.subplot(111, projection = '3d')
plt.title("3Ddata")
# 转换为3d数据，用于画在图表上，便于观察
X_val = np.reshape(outputsVal[:, 0], (10, 10))
Y_val = np.reshape(outputsVal[:, 1], (10, 10))
Z_val = np.reshape(outputsVal[:, 2], (10, 10))
ax.scatter(X_val, Y_val, Z_val)
ax.set_xlabel("x label", color='r')
ax.set_ylabel("y label", color='g')
ax.set_zlabel("z label", color='b')
plt.show()
'''
#np.savez("DATA.npz",dataSet,featuresSet,data_fft,dataLable)
#D=np.load("files.npz")

metrics_metrix=(-1*metrics.pairwise.pairwise_distances(X)).astype(np.int32)
metrics_metrix+=-1*metrics_metrix.min()
##设置谱聚类函数
n_clusters_= 5
lables = spectral_clustering(metrics_metrix,n_clusters=n_clusters_)
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    ##根据lables中的值是否等于k，重新组成一个True、False的数组
    my_members = lables == k
    ##X[my_members, 0] 取出my_members对应位置为True的值的横坐标
    plt.plot(X[my_members, 0], X[my_members, ], col + '.')
    
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
k=5
cluster=lables
cluster=cluster.reshape(k,1200)
counter = np.zeros(shape=(1200,k))
for ci in range(len(cluster)):
    for cj in range(k):
        indices_zero = np.nonzero(cluster[ci,:] == cj)
        T=np.int_(indices_zero[0])
        counter[ci,cj]=len(T)
right_numberx=0
for ri in range(len(counter)):
    tp = np.argmax(counter[ri,:])
    right_numberx=counter[ri,tp]+right_numberx

rightx=right_numberx/len(X)
