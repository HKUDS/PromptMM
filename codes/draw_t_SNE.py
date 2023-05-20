#自己代码的版本
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.ticker as ticker
import pickle
import datetime

from sklearn import datasets
from sklearn.manifold import TSNE

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


#---------------------------------------------------------------------------------------------------------------------------------------
import argparse
from math import log
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dataloader


import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, AIFBDataset, MUTAGDataset, BGSDataset

# from gcn import GCN
# from gcn_classify_linkprediction import GCN
#from gcn_mp import GCN
#from gcn_spmv import GCN
# import EWC

import scipy
from scipy.sparse import csr_matrix
import pickle
import random
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise
import datetime
#---------------------------------------------------------------------------------------------------------------------------------------











dataset='tiktok/'
# dataset='IJCAI_15'
model = 'gcn'  #target_feat, gcn, gat, graphsage

if dataset == 'imdb':
    n_classes = 3
    target_id = [0, 4278]
elif dataset == 'acm':
    n_classes = 3
    target_id = [0, 3025]
elif dataset == 'dblp':
    n_classes = 4
    target_id = [0, 2957]



#'red', 'cyan', 'blue', 'green', 'black', 'magenta', 'pink', 'purple','chocolate', 'orange', 'steelblue', 'crimson', 'lightgreen', 'salmon','gold', 'darkred'
#'x', 's', '^', 'o', 'v', '^', '<', '>', 'D'
# c = ['red', 'cyan', 'blue', 'black', 'green']  #Tmall
c = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple' , 'black', 'magenta', 'chocolate', ]  #retailrocket
# c = ['#444693', '#5c7a29', '#fcaf17', '#b3424a']  #retailrocket
c = ['red', 'yellow', 'green', 'blue', 'purple' , 'black', 'magenta', 'chocolate', ]  #retailrocket
c = ['#8F7D80', '#7A8FA1', '#7F92B3', '#b2a06e ', '#' , '#', '#', '#', ]  #retailrocket
c = ['#719758', '#5E5EA2', '#A25E5E', '#CC8F33', '#' , '#', '#', '#', ]  #retailrocket
m = ['x', 's', '^', 'o', 'v', '^', '<', '>', 'D']


#--------cora--------------------------------------------------------------------------------------------------------
# loadPath = "/home/ww/Code/LIB/dgl/examples/pytorch/gcn/Model/cora/first_time_to_try_this_model_For_savetime_matplotlib_cora_2021_10_02__09_01_12_n_hidden_128.pth"
# loadPath_No_SSL = ""
# raw_feature_path = "/home/ww/Code/LIB/dgl/examples/pytorch/gcn/data/cora/feature"
raw_feature_path = '/home/ww/FILE_from_ubuntu18/Code/work7/t_SNE/' + dataset + model  #gcn, gat, graphsage
# loadPath = "/home/ww/Code/LIB/dgl/examples/pytorch/gcn/Model/cora/cora_2021_10_06__12_27_56_first_time_to_try_ours_For_savetime_matplotlib_best_acc_0.8480000495910645_best_nmi_0.3806502933018018_best_epoch_193.pth"
loadPath = "/home/ww/FILE_from_ubuntu18/Code/work7/t_SNE/" + dataset + 'label_list'
# 应该就是一致性正则的结果:  /home/ww/Code/LIB/dgl/examples/pytorch/gcn/Model/cora/first_time_to_try_ours_For_savetime_matplotlib_cora_2021_10_03__11_35_45.pth

#--------cora--------------------------------------------------------------------------------------------------------


# #--------citeseer--------------------------------------------------------------------------------------------------------
# raw_feature_path = "/home/ww/Code/LIB/dgl/examples/pytorch/gcn/data/citeseer/feature"
# loadPath_SSL = ""
# loadPath_No_SSL = ""
# raw_feature_path = "/home/ww/Code/LIB/dgl/examples/pytorch/gcn/data/citeseer/feature"
# loadPath = ""
# #--------citeseer--------------------------------------------------------------------------------------------------------


# #--------pubmed--------------------------------------------------------------------------------------------------------
# loadPath_SSL = ""
# loadPath_No_SSL = ""
# #--------pubmed--------------------------------------------------------------------------------------------------------


# checkpoint= torch.load(loadPath)
input_label = pickle.load(open(loadPath, 'rb'))
input_data = pickle.load(open(raw_feature_path, 'rb'))
# input_data = input_data.todense()
input_data = input_data.cpu().detach().numpy()
# model = checkpoint['model']
# model_NOSSL = checkpoint_NOSSL['model']
# model_SSL = checkpoint_SSL['model']
# params = model.state_dict()


def get_data(embedding, beh_index, test_index):

    data = embedding[test_index].detach().numpy()
    # data = np.array(embedding)                                                 #[1797, 64]
    label = np.empty(data.shape[0], dtype=np.int32)
    label.fill(beh_index)       #[1797]
    n_samples, n_features = embedding.shape                                        #1797, 64
    return data, label, n_samples, n_features



# 对样本进行预处理并画图
def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)     # 对数据进行归一化处理
    fig = plt.figure()      # 创建图形实例
    ax = plt.subplot(111)       # 创建子图
    ax.axis('off')
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        # plt.text(data[i, 0], data[i, 1], str(label[i]/10), color=plt.cm.Set1(label[i] / 10),
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=c[label[i]],
                 fontdict={'weight': 'bold', 'size': 8})
    plt.xticks()        # 指定坐标的刻度
    plt.yticks()
    # plt.title(title, fontsize=11, y=-0.13,verticalalignment='bottom' , horizontalalignment='center')
    # 返回值
    return fig


def plot_embedding_3D(data, label, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = len(test_user)
    # plt.rcParams["le"]

    for i in range(data.shape[0]):
        # xs = np.arange(data.shape[1]) 
        # ys = np.arange(data.shape[1])
        xs = data[i].swapaxes(0,1)[0] 
        ys = data[i].swapaxes(0,1)[1]
        zs = data[i].swapaxes(0,1)[2]
        ax.scatter(xs, ys, zs, c=c[i], alpha=0.1, s=0.5)
    ax.set_xlabel('X Label')
    ax.set_xlabel('Y Label')
    ax.set_xlabel('Z Label')
    return fig


def get_input():
    input_data = checkpoint['node_embed'].cpu()
    input_label = checkpoint['labels'].cpu().numpy()
    input_pre_labels = checkpoint['pre_labels'].cpu()
    input_pre_labels = torch.argmax(input_pre_labels, dim=-1).numpy()
  
    # return input_data, input_label, input_pre_labels
    return input_data, input_label, input_pre_labels


#get_input_raw_data
def get_input_raw_data():
    input_data = raw_feature
    # input_data = checkpoint['node_embed'].cpu()
    input_label = checkpoint['labels'].cpu().numpy()
    input_pre_labels = checkpoint['pre_labels'].cpu()
    input_pre_labels = torch.argmax(input_pre_labels, dim=-1).numpy()
  
    # return input_data, input_label, input_pre_labels
    return input_data, input_label, input_pre_labels


# def get_input_3D():
#     input_data = [None]*len(behaviors)
#     input_label = [None]*len(behaviors)
    
#     for i in range(len(behaviors)):
#         if i == (len(behaviors)-1):
#             input_data[i], input_label[i] , n_samples, n_features = get_data(checkpoint_SSL['user_embed'].cpu(), i, test_user)        #[1797, 64]    [1797]    1797   64
        
#         else:
#             input_data[i], input_label[i] , n_samples, n_features = get_data(checkpoint_SSL['user_embeds'][i].cpu(), i, test_user)


#         # print(f"data.shape: {data.shape}")
#         print(f"input_data.shape: {input_data[i].shape}")
#         # print(f"label.shape: {input_data[i].shape}")
#         print(f"input_label.shape: {input_label[i].shape}")
#     input_data = np.array(input_data)
#     input_label = np.array(input_label)
#     return input_data, input_label 




# # #--------------------------------------------------------------------------------------------------------
# args = checkpoint['args']

# input_data, input_label, input_pre_labels = get_input_raw_data()

# input_data, input_label, input_pre_labels = get_input()
time = datetime.datetime.now()
print("Starting compute t-SNE Embedding...", time)
# ts = TSNE(n_components=3, init='pca', random_state=0)
# ts = TSNE(n_components=2, init='pca', random_state=0)
ts = TSNE(n_components=2, init='pca')

# t-SNE降维
result = ts.fit_transform(input_data)  #[1797, 2]
time = datetime.datetime.now()
print("Starting compute t-SNE Embedding...", time)

# pickle.dump(result, open('/home/ww/Code/LIB/dgl/examples/pytorch/gcn/TSNE/cora/first time to try this model_For_savetime_matplotlib_cora_2021_10_02__08_41_42_n-hidden_128.pth', 'wb'))  #-----------------------------
# pickle.dump(result, open('/home/ww/Code/work3/BSTRec/TSNE/'+'_2D_no_meta_no_SSL_all_beh', 'wb'))
# # #--------------------------------------------------------------------------------------------------------


# #--------------------------------------------------------------------------------------------------------
# ts = [None]*len(behaviors)
# result = [None]*len(behaviors)

# input_data, input_label = get_input_3D()
# print('Starting compute t-SNE Embedding...')
# for i in range(input_data.shape[0]):
#     ts[i]= TSNE(n_components=3, init='pca', random_state=0)
#     # t-SNE降维
#     result[i] = ts[i].fit_transform(input_data[i])  #[1797, 2]
# pickle.dump(result, open('/home/ww/Code/work3/T_Meta_SSL_MBGCN/TSNE/'+dataset,'wb'))
# print('Starting compute t-SNE Embedding...')
# #--------------------------------------------------------------------------------------------------------


# /home/ww/Code/work3/BSTRec/TSNE/IJCAI_15
# result = pickle.load(open('/home/ww/Code/work3/BSTRec/TSNE/'+ dataset + '/'+dataset+'_2D_no_meta_no_SSL_all_beh','rb'))  #---------------------------
# result = pickle.load(open('/home/ww/Code/work3/BSTRec/TSNE/'+ dataset + '/'+dataset+'_2D_no_meta_SSL_all_beh','rb'))
# result = pickle.load(open('/home/ww/Code/work3/BSTRec/TSNE/IJCAI_15_2D_no_meta_SSL','rb'))



# pickle.dump(result, open('/home/ww/Code/work3/BSTRec/TSNE/'+dataset+'_2D_no_meta_SSL_all_beh', 'wb'))  #-----------------------------
# pickle.dump(result, open('/home/ww/Code/work3/BSTRec/TSNE/'+dataset+'_2D_no_meta_no_SSL_all_beh', 'wb'))


# # 调用函数，绘制图像

fig = plot_embedding(result, input_label, 'gcn_cora')   #The t-SNE visualization of embedding on Tmall.
# 调用函数，绘制图像
# fig = plot_embedding_3D(np.array(result), input_label, 't-SNE Embedding of digits')   #

# plt.savefig('/home/ww/Code/work3/BSTRec/TSNE/IJCAI_SSL.pdf')
# 显示图像
plt.show()





# fig.colorbar(cax)
plt.savefig('/home/ww/FILE_from_ubuntu18/Code/work7/t_SNE/' + dataset + model +'.pdf')  #target_feat, gcn, gat, graphsage

# '/home/ww/FILE_from_ubuntu18/Code/work7/t_SNE/' + dataset + 'target_feat'  #gcn, gat, graphsage

# plt.savefig('D:\CODE\master_behavior_attention\Picture\self_attentionself_attention6.jpg')
# plt.savefig('D:\CODE\master_behavior_attention\Pictureattentionself_attention.jpg')
# plt.show()
# PlotMats(self_attention_ndarray = params['self_attention_para'].cpu().numpy(), , show=False, savePath='visualization/legend.pdf', vrange=[0, 1])


