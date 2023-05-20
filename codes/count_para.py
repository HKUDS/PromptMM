import torchsummary
from Models import Teacher_Model, Student_LightGCN, Student_GCN, Student_MLP, PromptLearner
import torch
import numpy as np
import scipy.sparse as sp
import pickle
import torch.nn as nn


from sklearn.decomposition import PCA, FastICA
from sklearn import manifold
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# from utility.parser import parse_args
# args = parse_args()
from utility.parser import args, select_dataset

select_dataset()



# class PromptLearner(nn.Module):
#     def __init__(self, image_feats=None, text_feats=None, ui_graph=None):
#         super().__init__()
#         self.ui_graph = ui_graph
#         if args.hard_token_type=='pca':
#             hard_token_image = PCA(n_components=args.embed_size).fit_transform(image_feats)
#             hard_token_text = PCA(n_components=args.embed_size).fit_transform(text_feats)
#         elif args.hard_token_type=='ica':
#             hard_token_image = FastICA(n_components=args.embed_size, random_state=12).fit_transform(image_feats)
#             hard_token_text = FastICA(n_components=args.embed_size, random_state=12).fit_transform(text_feats)
#         elif args.hard_token_type=='isomap':
#             hard_token_image = manifold.Isomap(n_neighbors=5, n_components=args.embed_size, n_jobs=-1).fit_transform(image_feats)
#             hard_token_text = manifold.Isomap(n_neighbors=5, n_components=args.embed_size, n_jobs=-1).fit_transform(text_feats)
#         # elif args.hard_token_type=='tsne':
#         #     hard_token_image = TSNE(n_components=args.embed_size, n_iter=300).fit_transform(image_feats)
#         #     hard_token_text = TSNE(n_components=args.embed_size, n_iter=300).fit_transform(text_feats)
#         # elif args.hard_token_type=='lda':
#         #     hard_token_image = LinearDiscriminantAnalysis(n_components=args.embed_size).fit_transform(image_feats)
#         #     hard_token_text = LinearDiscriminantAnalysis(n_components=args.embed_size).fit_transform(text_feats)

#         self.item_hard_token = nn.Embedding.from_pretrained(torch.mean((torch.stack((torch.tensor(hard_token_image).float(), torch.tensor(hard_token_text).float()))), dim=0), freeze=False).weight
#         self.user_hard_token = nn.Embedding.from_pretrained(torch.mm(ui_graph, self.item_hard_token), freeze=False).weight

#         # self.gnn_trans_user =  nn.Linear(args.embed_size, args.embed_size)
#         # self.gnn_trans_item =  nn.Linear(args.embed_size, args.embed_size)
#         # nn.init.xavier_uniform_(self.gnn_trans_user.weight) 
#         # nn.init.xavier_uniform_(self.gnn_trans_item.weight) 
#         # self.gnn_trans_user = self.gnn_trans_user.cuda() 
#         # self.gnn_trans_item = self.gnn_trans_item.cuda() 
#         # self.item_hard_token = torch.mean((torch.stack((torch.tensor(hard_token_image).float(), torch.tensor(hard_token_text).float()))), dim=0).cuda()


#     def forward(self):
#         # self.user_hard_token = self.gnn_trans_user(torch.mm(self.ui_graph, self.item_hard_token))
#         # self.item_hard_token = self.gnn_trans_item(self.item_hard_token)

#         return self.user_hard_token , self.item_hard_token


print(torchsummary.__file__)

def matrix_to_tensor(cur_matrix):
    if type(cur_matrix) != sp.coo_matrix:
        cur_matrix = cur_matrix.tocoo()  #
    indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
    values = torch.from_numpy(cur_matrix.data)  #
    shape = torch.Size(cur_matrix.shape)

    return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()  #


image_feats = np.load(args.data_path + '{}/image_feat.npy'.format(args.dataset))
text_feats = np.load(args.data_path + '{}/text_feat.npy'.format(args.dataset))

ui_graph = ui_graph_raw = pickle.load(open(args.data_path + args.dataset + '/train_mat','rb'))
# ui_graph = csr_norm(ui_graph, mean_flag=True)
ui_graph = matrix_to_tensor(ui_graph)



device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")  #
# techaer_model = Teacher_Model(ui_graph.shape[0], ui_graph.shape[1], 32, [64, 64], [0.1, 0.1], image_feats, text_feats).to(device).cuda()
teacher_model = Teacher_Model(ui_graph.shape[0], ui_graph.shape[0], args.embed_size, eval(args.weight_size), args.mess_dropout, image_feats, text_feats)      


if args.student_model_type=='lightgcn':
    student_model = Student_LightGCN(ui_graph.shape[0], ui_graph.shape[0], args.student_embed_size, args.student_n_layers, args.mess_dropout, image_feats, text_feats)   
    # self.student_model.init_user_item_embed(self.u_final_embed, self.i_final_embed)
elif args.student_model_type=='gcn': 
    student_model = Student_GCN(ui_graph.shape[0], ui_graph.shape[1], args.student_embed_size, args.student_n_layers, args.mess_dropout, image_feats, text_feats)   
elif args.student_model_type=='mlp': 
    student_model = Student_MLP()   
    # self.student_model.init_user_item_embed(self.u_final_embed, self.i_final_embed)

prompt_module = PromptLearner(image_feats, text_feats, ui_graph).cuda()


# torchsummary.summary(model, [(ui_graph.shape[0], ui_graph.shape[1]), (ui_graph.shape[1], ui_graph.shape[0])])

# import torchvision.models as models
# from torchstat import stat

# # test_model = models.resnet18()
# stat(model, [(35598, 18357), (18357, 35598)])

# n_parameters = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
# print(n_parameters)


# n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(n_parameters)


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

model_structure(teacher_model)

