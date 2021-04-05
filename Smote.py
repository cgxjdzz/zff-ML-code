import pandas as pd
import numpy as np
import warnings
from sklearn.neighbors import NearestNeighbors
import random
import numpy as np
from math import floor

#怎么评估增加后的数据呢，数据聚集程度如何来看
def Smote(feature,y,zero_strategy,one_strategy,seed):
    data=pd.concat([y,feature],axis = 1,join = 'inner')
    #先加一块

    random.seed(seed)
    #设立伪随机数种子

    grouped = data.groupby('Label')
    y0 = grouped.get_group(0)
    y1 = grouped.get_group(1)
    X0 = pd.DataFrame(feature,index=y0.index)
    X1 = pd.DataFrame(feature, index=y1.index)
    data0 = pd.DataFrame(data,index=y0.index)
    data1 = pd.DataFrame(data,index=y1.index)
    #按标签分成两组

    X0_new=X0
    X1_new=X1
    #准备传出去的玩意


    if data0.shape[0]>zero_strategy or data1.shape[0]>one_strategy:
        warnings.warn('strategy data less than original data',ValueError)


    k0=round(((zero_strategy-data0.shape[0])/data0.shape[0]+1)*2)
    k1=round(((one_strategy-data1.shape[0])/data1.shape[0]+1)*2)
    #或者由传入参数决定

    X0_val=X0.values
    nbrs = NearestNeighbors(n_neighbors=k0, algorithm='auto').fit(X0_val)
    distances, indices = nbrs.kneighbors(X0_val)
    # indices 是二维数组，第n行为第n特征的临近点，但indices[n][0]固定为n，可以忽略

    #调试代码可忽略
    The_new0=pd.DataFrame()
    The_new1=pd.DataFrame()
    # 调试代码


    #遍历data0中每个样本
    while 1:

        i0=round(random.random()*(X0.shape[0]-1))

        if X0_new.shape[0]>=zero_strategy:
            break
        #先确认一下是否达到目标数量

        seed += 1
        random.seed(seed)
        # 重设一下seed，要不都重复了，或者可以直接整个函数，有点麻烦这样
        Nbr_ind=indices[i0][floor(random.random()*(k0-1))+1]
        #随机取这个点的邻近样本
        Nbr=X0_val[Nbr_ind]

        seed+=1
        random.seed(seed)
        #重设一下seed，要不都重复了，或者可以直接整个函数，有点麻烦这样
        New=(Nbr-X0_val[i0])*random.random()+X0_val[i0]
        New=pd.DataFrame(data=[New],columns=feature.columns)
        Try_data=feature.append(New,ignore_index=True)
        #将新建的加入末尾，并再次寻找最近点
        Try_nbrs = NearestNeighbors(n_neighbors=4, algorithm='auto').fit(Try_data)
        Try_distances,Try_indices=Try_nbrs.kneighbors(New)


        temp = 0
        # 这个数一下他邻近和他不同的点
        for j0 in [1,2,3]:
            if y.loc[Try_indices[0][j0]][0]==1:
                #取邻近点的标签。。。写的有点麻烦但是这个意思
                temp+=1

        if temp<3:
            X0_new= X0_new.append(New,ignore_index=True)
            # 调试代码#调试代码
            The_new0 = The_new0.append(New, ignore_index=True)





    #由于太菜就把这玩意再抄一遍，但如果考虑降低代码重复率的话，可以把这一堆写成函数
    X1_val = X1.values
    nbrs = NearestNeighbors(n_neighbors=k1, algorithm='auto').fit(X1_val)
    distances, indices = nbrs.kneighbors(X1_val)
    # indices 是二维数组，第n行为第n特征的临近点，但indices[n][0]固定为n，可以忽略

    # 遍历data0中每个样本
    while 1:

        i1 = round(random.random() * (X1.shape[0] - 1))

        if X1_new.shape[0] >= one_strategy:
            break
        # 先确认一下是否达到目标数量

        seed += 1
        random.seed(seed)
        # 重设一下seed，要不都重复了，或者可以直接整个函数，有点麻烦这样
        Nbr_ind = indices[i1][floor(random.random() * (k1 - 1)) + 1]
        # 随机取这个点的邻近样本
        Nbr = X1_val[Nbr_ind]

        seed += 1
        random.seed(seed)
        # 重设一下seed，要不都重复了，或者可以直接整个函数，有点麻烦这样
        New = (Nbr - X1_val[i1]) * random.random() + X1_val[i1]
        New = pd.DataFrame(data=[New],columns=feature.columns)
        Try_data = feature.append(New, ignore_index=True)
        # 将新建的加入末尾，并再次寻找最近点
        Try_nbrs = NearestNeighbors(n_neighbors=4, algorithm='auto').fit(Try_data)
        Try_distances, Try_indices = Try_nbrs.kneighbors(New)

        temp = 0
        # 这个数一下他邻近和他不同的点
        for j1 in [1, 2, 3]:
            if y.loc[Try_indices[0][j1]][0] == 0:
                # 取邻近点的标签。。。写的有点麻烦但是这个意思
                temp += 1

        if temp < 3:
            X1_new = X1_new.append(New, ignore_index=True)


            #调试代码
            The_new1=The_new1.append(New,ignore_index=True)


    return X0_new,X1_new
