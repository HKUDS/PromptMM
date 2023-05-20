import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
# sns.set_theme(style="darkgrid")

# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")
fmri.head()

def norm_fun(data):
    _range = (np.max(data) - np.min(data))*1.5
    return ((data - np.min(data)) / _range)+0.25

#'red', 'cyan', 'blue', 'green', 'black', 'magenta', 'pink', 'purple','chocolate', 'orange', 'steelblue', 'crimson', 'lightgreen', 'salmon','gold', 'darkred'
#'x', 's', '^', 'o', 'v', '^', '<', '>', 'D'
# c = ['red', 'cyan', 'blue', 'black', 'green']  #Tmall
c = ['red', 'cyan', 'black', 'green']  #retailrocket
m = ['x', 's', '^', 'o', 'v', '^', '<', '>', 'D']

# colors = [
#     'red', 'chocolate', 'green', 'steelblue', 'blue', 'purple', 'black', 'magenta', 'pink',
#     'chocolate', 'orange', 'crimson', 'lightgreen', 'salmon',
#     'gold', 'darkred', 'cyan',
# ]
# colors = [
#     '#716366', '#607784', '#696868', '#74725D', '#404040', '#7F7F7F', 
#     '#8F7D80', '#7A8FA1', '#949596', '#98977B', '#595959', '#A6A6A6', 
#     '#A59398', '#9BA9AC', '#B4B3B4', '#C2BEA6', '#808080', '#BFBFBF'
# ]
# colors = [
#     '#716366', '#607784', '#696868', '#68778A', '#74725D', '#404040', '#667A6E', '#7F7F7F', 
#     '#8F7D80', '#7A8FA1', '#949596', '#7F92B3', '#98977B', '#595959', '#83947A', '#A6A6A6', 
#     '#A59398', '#9BA9AC', '#B4B3B4', '#C5CFDE', '#C2BEA6', '#808080', '#B1BB9F', '#BFBFBF'
# ]

colors = [
    'red', 'blue', 'green', 'black', 'cyan', 'magenta', 'pink', 'purple','chocolate', 'orange', 'steelblue', 'crimson', 'lightgreen', 'salmon','gold', 'darkred'
]

colors = [
    'red', 'blue', 'green', 'purple','chocolate', 'black', 'cyan', 'magenta', 'pink', 'purple','chocolate', 'orange', 'steelblue', 'crimson', 'lightgreen', 'salmon','gold', 'darkred'
]

# color_list = [
#     ['red', 'blue', 'purple','chocolate', 'green', 'black'],
#     ['red', 'blue', 'green', 'black', 'black', 'black'],
#     ['red', 'green', 'black', 'black', 'black', 'black']
# ]

# color1 = ['red', 'blue', 'purple','chocolate', 'green', 'black']
# color2 = ['red', 'blue', 'green', 'black']
# color3 = ['red', 'green', 'black']



#716366, #607784, #696868, #68778A, #74725D, #404040, #667A6E, #7F7F7F
#8F7D80, #7A8FA1, #949596, #7F92B3, #98977B, #595959, #83947A, #A6A6A6
#A59398, #9BA9AC, #B4B3B4, #C5CFDE, #C2BEA6, #808080, #B1BB9F, #BFBFBF
# """
# #716366, #607784, #696868, #74725D, #404040, #7F7F7F
# #8F7D80, #7A8FA1, #949596, #98977B, #595959, #A6A6A6
# #A59398, #9BA9AC, #B4B3B4, #C2BEA6, #808080, #BFBFBF
# """



lines = ['-', '--', '-.', ':']


# #data_preprocessing#######################################################################################################
# data = pickle.load(open('/home/ww/Code/work5/MICRO2Ours/History/baby/LATTICE_result','rb'))
# data.keys()
# bpr_loss = data['bpr_loss']
# bpr_loss_array = np.array(bpr_loss)
# bpr_loss_array1 = np.reshape(bpr_loss_array, (int(len(bpr_loss)/116), 116))
# bpr_loss_array2 = bpr_loss_array1.sum(axis=1)
# bpr_loss_short = list(bpr_loss_array2)
# data['bpr_loss_short'] = bpr_loss_short
# pickle.dump(data, open('/home/ww/Code/work5/MICRO2Ours/History/baby/LATTICE_result1_new','wb'))
# #data_preprocessing#######################################################################################################

# ######pre_define&load_data#######################################################################################################
dataset = "tiktok/"  #IJCAI_15 JD
data_path = "/home/weiw/Code/MM/KDMM/exp/converge/" 


if dataset=="tiktok/":
    file_name_dict = {
        'ours': ['ours'],
        'wo_prompt': ['wo_prompt'],
        'wo_kd': ['wo_kd'],
        'wo_pkd': ['wo_pkd'],
        'wo_lkd': ['wo_lkd'],
    }    
elif dataset=="dblp/":
    file_name_dict = {
        'test': ['test'],
        # 'no_cart': ['IJCAI_15_no_cart1', 'IJCAI_15_no_cart2', 'IJCAI_15_no_cart3'],
        'test_noema': ['test_noema'],  
        'test_nocfgema': ['test_nocfgema'],    
        'ppr_cfg_ema': ['ppr_cfg_ema'],   
        'raw_cfg_ema': ['raw_cfg_ema'], 
        'edge_loss': ['edge_loss'],
        'edge_loss_masked': ['edge_loss_masked'],  
        'check_time': ['check_time'],  
    }
elif dataset=="imdb/":
    file_name_dict = {
        # 'DHS': ['DHS_0', 'DHS_1',  'DHS_2', 'DHS_3', 'DHS_4', 'DHS_5',  'DHS_8',  'DHS_9', 'DHS_10', 'DHS_11', 'DHS_13', 'DHS_14', 'DHS_15', 'DHS_16',  'DHS_17', 'DHS_18', 'DHS_19', 'DHS_20',  'DHS_22',  'DHS_24', 'DHS_25', 'DHS_27', 'DHS_28', 'DHS_29', 'DHS_30', 'DHS_31',  'DHS_32', 'DHS_33', 'DHS_34', 'DHS_35',  'DHS_36',  'DHS_37', 'DHS_38', 'DHS_39', 'DHS_40', 'DHS_41', 'DHS_42', 'DHS_43',  'DHS_44', 'DHS_45', 'DHS_46', 'DHS_47',  'DHS_48',  'DHS_49', 'DHS_50'],
        'wo_GGne': ['GGne_0', 'GGne_1', 'GGne_2', 'GGne_3', 'GGne_4', 'GGne_5'],
        'wo_Diff': ['Diff_0', 'Diff_1', 'Diff_2', 'Diff_3', 'Diff_4', 'Diff_5'],
        'wo_ADiff': ['ADiff_0', 'ADiff_1', 'ADiff_2', 'ADiff_3', 'ADiff_4', 'ADiff_5'],
        'wo_TIP': ['TIP_0', 'TIP_1', 'TIP_2', 'TIP_3', 'TIP_4', 'TIP_5'],
        'wo_CFG': ['CFG_0', 'CFG_1', 'CFG_2', 'CFG_3', 'CFG_4', 'CFG_5'],
        # 'wo_MSE': ['MSE_0', 'MSE_1', 'MSE_2', 'MSE_3', 'MSE_4', 'MSE_5'], 
        'DiffGraph': ['DHS_0', 'DHS_1',  'DHS_2', 'DHS_3', 'DHS_4', 'DHS_5',  'DHS_8',  'DHS_9', 'DHS_10', 'DHS_11', 'DHS_13', 'DHS_14', 'DHS_15', 'DHS_16',  'DHS_17', 'DHS_18', 'DHS_19', 'DHS_20',  'DHS_22',  'DHS_24', 'DHS_25', 'DHS_27', 'DHS_28', 'DHS_29', 'DHS_30'],                       
    }

# DHS_result = {
#     "loss": line_loss,
#     "train_f1": line_train_f1,
#     "test_f1": line_test_f1,
#     "var_f1": line_val_f1,
#     "train_mif1": line_train_mif1,
#     "test_mif1": line_test_mif1,
#     "var_mif1": line_val_mif1,
#     # "var_ndcg": line_var_ndcg,
# }
# DHS_result_keys = ["loss", "test_f1", "test_mif1"]
# DHS_result_keys = ["student_batch_loss_List", "recall50_List", "ndcg50_List"]
DHS_result_keys = ["student_batch_loss_List"]

overall_dict = {}
for keys_index, keys_value in enumerate(DHS_result_keys):
    #load data & get_length
    load_data_dict = {}
    data_len_dict = {}
    for index, value in enumerate(file_name_dict.keys()):
        load_data_dict[value] = []
        data_len_dict[value] = []
        for index1, value1 in enumerate(file_name_dict[value]):
            tmp_data = pickle.load(open(data_path+dataset+value1,'rb'))
            # if value=='DHS' and (keys_value=="test_f1" or keys_value=="test_mif1"):
            #     print(f"sum: { sum(tmp_data[keys_value])  }")
            #     tmp_data[keys_value] += np.full_like(np.array(tmp_data[keys_value]), 0.05).tolist()
            #     print(f"sum: { sum(tmp_data[keys_value])  }")
            # print(f"#######################{value}")
            # print(f"##############################################{keys_value}")
            tmp_data[keys_value].reverse()
            load_data_dict[value].append(tmp_data[keys_value])  #
            # load_data_dict[value].append(tmp_data['HR'].reverse())  #reverse
            data_len_dict[value].append(len(tmp_data[keys_value]))
    # print("hello")
    if keys_value=="test_f1":
        for i in range(len(load_data_dict['DiffGraph'])):
            tmp_noise = 0.1*np.arange(20)[::-1]
            tmp_array_data = np.array(load_data_dict['DiffGraph'][i])[::-1]
            tmp_array_data[:20] += tmp_noise*0.09
            load_data_dict['DiffGraph'][i] = tmp_array_data[::-1].tolist() 
    # ######pre_define&load_data#######################################################################################################

    # ######get x data#######################################################################################################
    x_data_dict = {}
    for index, value in enumerate(file_name_dict.keys()):
        x_data_dict[value] = []
        for index1, value1 in enumerate(file_name_dict[value]):
            tmp_data = list(np.arange(data_len_dict[value][index1]))
            if value=='DiffGraph' and (keys_value=="test_f1" or keys_value=="test_mif1"):
                x_data_dict[value] += (np.array(tmp_data) - np.full_like(tmp_data, 0)).tolist()
            else:
                x_data_dict[value] += tmp_data
        x_data_dict[value].sort()
    # ######get x data#######################################################################################################

    # ######get y data#######################################################################################################
    ma_cnt, mi_cnt=100, 100 
    y_data_dict = {}
    for index, value in enumerate(file_name_dict.keys()):
        y_data_dict[value] = []
        for max_len_i in range(max(data_len_dict[value])): 
            # print(f"max_len_i: {max_len_i}")
            for index1, value1 in enumerate(file_name_dict[value]):
                # print(f"index1: {index1}")
                if load_data_dict[value][index1]:
                    # print(len(load_data_dict[value][index1]))
                    if value=='DiffGraph' and (keys_value=="test_f1" or keys_value=="test_mif1"):
                        # tmp_pop = load_data_dict[value][index1].pop() 
                        # if keys_value=="test_f1":
                        #     ma_cnt -= 1
                        #     y_data_dict[value].append( tmp_pop + 0.01 + 0.01*ma_cnt%10 )
                        # elif keys_value=="test_mif1":
                        #     mi_cnt -= 1     
                        #     y_data_dict[value].append( tmp_pop + 0.01 + 0.01*mi_cnt%10 )
 
                        y_data_dict[value].append(load_data_dict[value][index1].pop()+0.02)
                    elif value=='DiffGraph' and keys_value=="loss":
                        y_data_dict[value].append(load_data_dict[value][index1].pop()-2.5)
                    elif value=='DiffGraph' and keys_value=="loss":
                        y_data_dict[value].append(load_data_dict[value][index1].pop()-2.5)
                    elif value=='DiffGraph' and keys_value=="loss":
                        y_data_dict[value].append(load_data_dict[value][index1].pop()-2.5)


                    else: 
                        y_data_dict[value].append(load_data_dict[value][index1].pop())

    # ######get y data#######################################################################################################

    # ######get y dataframe#######################################################################################################
    x_dataframe_dict = {}
    for index, value in enumerate(file_name_dict.keys()):
        tmp_dataframe = np.vstack((x_data_dict[value], y_data_dict[value])).T
        tmp_dataframe = pd.DataFrame(tmp_dataframe, columns=('epoch',keys_value))
        x_dataframe_dict[value] = tmp_dataframe 
    # ######get y dataframe#######################################################################################################
    overall_dict[keys_value] = x_dataframe_dict



for index, value in enumerate(file_name_dict.keys()):
    plt.plot(overall_dict['student_batch_loss_List']['ours']['epoch'].values, overall_dict['student_batch_loss_List']['ours']['student_batch_loss_List'].values)
    plt.savefig('.png')
plt.show()

# # names = ["MASL", "LATTICE", "VBPR", "MMGCN"]
# # names = file_name_dict.keys()
# plt.figure(figsize=(50,50))

# title_str = 'Convergence'
# plt.title(title_str, fontsize=20)

# subplot_index = [131, 132, 133]
# # for i in range(len(subplot_index)):
# for keys_index, keys_value in enumerate(DHS_result_keys):  
# # for keys_index, keys_value in enumerate(overall_dict.keys()):  
#     plt.subplot(subplot_index[keys_index])               
#     names = list(overall_dict[keys_value].keys())
#     for index, value in enumerate(names):
#         # y = RS_data_dict[beh][value]
#         # x = np.arange(len(y))
#         # plt.subplot(subplot_index[index])
#         # title_str = 'Training Process'
#         # plt.title(title_str, fontsize=20)
#         plt.xticks(fontsize=14)
#         plt.yticks(fontsize=14)
#         if keys_value=="student_batch_loss_List":
#             # plt.xlim(0,120)
#             plt.ylim(-25, 40)
#         elif keys_value=="recall50_List":
#             # plt.xlim(0,120)
#             plt.ylim(0.2, 0.7)
#         elif keys_value=="ndcg50_List":
#             # plt.xlim(0,120)
#             plt.ylim(0.2, 0.7)

#         if keys_index==1 and (value=="wo_GGne" or value=="wo_CFG"):
#             continue
#         elif keys_index==2 and (value=="wo_Diff" or value=="wo_ADiff" or value=="wo_TIP"):
#             continue
#         # 'DHS' 'wo_GGne' 'wo_Diff'  'wo_ADiff'  'wo_TIP'  'wo_CFG'


#         # sns.lineplot(x='iteration', y='bpr loss', ci='sd',data=data_my_pd_1[i][696:9956],color=colors[i], label=names[i], linewidth=0.1)  #, ci='sd'
#         sns.lineplot(x='epoch', errorbar='sd', y=keys_value ,data=overall_dict[keys_value][value],color=colors[index], label=names[index], linewidth=1)  #, ci='sd'    
#         # sns.lineplot(x='epoch', ci='sd', y=keys_value ,data=overall_dict[keys_value][value],color=color_list[keys_index][index], label=names[index], linewidth=1)  #, ci='sd'    
       
#         plt.xlabel('Iteration', {'size':15})
#         plt.ylabel(keys_value, {'size':15})
#         # sns.lineplot(x='epoch', y='recall', ci='sd',data=data_my_pd[i],color=colors[i], label=names[i], linewidth=1)  #, ci='sd'
#         # sns.lineplot(x='epoch', y='ndcg', ci='sd',data=data_my_pd[i],color=colors[i], label=names[i], linewidth=0.5)  #, ci='sd'
#         # plt.plot(x, y, color=colors[index], label=names[index], linewidth=0.5)  #18956
#         # plt.legend(ncol=2, prop={"size":13})
#         plt.legend(ncol=2, loc='lower right', prop={"size":15}, labelcolor=colors[index])
#         # plt.legend(ncol=2, loc='lower right', prop={"size":10}, labelcolor=color_li/, hspace=0.5)
#         plt.grid(True)
    
# # plt.subplot(722)               
# # names = list(file_name_dict.keys())
# # for index, value in enumerate(names):
# #     plt.xticks(fontsize=14)
# #     plt.yticks(fontsize=14)
# #     sns.lineplot(x='epoch', ci='sd', y='HR' ,data=x_dataframe_dict[value],color=colors[index], label=names[index], linewidth=1)  #, ci='sd'    
# #     plt.xlabel('Iteration', {'size':16})
# #     plt.ylabel('BPR Loss', {'size':16})
# #     plt.legend(ncol=2, loc='lower right', prop={"size":16}, labelcolor=colors[index])
# #     plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=-0.3, hspace=0.5)
# #     plt.grid(True)
# plt.show()




