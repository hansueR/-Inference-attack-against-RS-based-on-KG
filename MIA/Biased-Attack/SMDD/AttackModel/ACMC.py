import os

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# debug
# import pydevd_pycharm
# pydevd_pycharm.settrace('172.25.231.221', port=3931, stdoutToServer=True, stderrToServer=True)

num_latent = 100

def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(num_latent, 32)
#         self.fc2 = nn.Linear(32, 16)
#         self.fc3 = nn.Linear(16, 2)

#     def forward(self, x):
#         x = x.view(-1, num_latent)
#         x = F.relu(self.fc1(x))
#         x = F.softmax(self.fc3(x))
#         return x

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_latent, 32)
        self.fc2 = nn.Linear(32, 16)  # 更新输出大小以匹配下一层
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = x.view(-1, num_latent)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # 对第二层使用ReLU激活函数
        x = F.softmax(self.fc3(x),dim=1)
        return x


topk = 100
# pathS = "../../../Recommender/caser_pytorch-master/datasets"
pathS = "/content/drive/MyDrive/MIA/Recommender/caser_pytorch-master/datasets"
pathT = pathS
# Vpath = "../../../Dataset/Vectorized_itemEmbed"
Vpath = "/content/drive/MyDrive/MIA/Dataset/Vectorized_itemEmbed"


# ------------------shadow------------------

fr_vector_shadow = open(Vpath + "/book_itemMatrix.txt", 'r')
path = "/content/drive/MyDrive/MIA/Dataset/data/processed_book/"
itemDict = {}
with open(path + "book_itemDict", 'r') as f:
    for line0 in f.readlines():
        line0 = line0.strip().split('\t')
        itemDict[int(line0[1])] = int(line0[0])

vectors1 = {}  # vectors for shadow items
index = -1
for line in fr_vector_shadow.readlines():
    index = index + 1
    line = line.split(' ')
    line = list(map(float, line))
    t_vectors = torch.tensor(line)
    if index in itemDict: # 重要
      vectors1[itemDict[index]] = t_vectors
# print(vectors1) # 正常
# read recommends
# Smember_rec = open("/content/drive/MyDrive/Recommendation-system-based-on-ripplenet/book_Smember_recommendations" , 'r')  
Smember_rec = open(pathS + "/book_Smember_recommendations", 'r') # no kg

recommend_Smember = {}
num = 0
for line in Smember_rec.readlines():
    line = line.split('\t')
    # # print("line:",line)
    # sessionID = line[0]
    # itemID = line[1]
    # print("sessionID:", sessionID)
    # print("itemID:", itemID)
    # recommend_Smember.setdefault(int(sessionID), []).append(int(itemID))

    sessionID = line[0]
    itemID = line[1]
    # 跳过itemID为'NONE'的行
    if itemID == 'NONE':
        continue
    # 将会话ID和项目ID添加到字典中（前提是它们是有效的整数）
    try:
        recommend_Smember.setdefault(int(sessionID), []).append(int(itemID))
    except ValueError as e:
        # 处理转换错误，例如打印错误消息
        # print("ValueError:", e)
        num = num + 1
        # 如果不想添加错误的行到字典中，可以继续跳过处理
        continue
print(num)
# print("length of recommend_Smember.keys:", len(recommend_Smember.keys()))
# print("recommend_Smember.keys:", sorted(recommend_Smember.keys()))

recommend_Snonmem = {}  # recommends for shadow
Snonmem_rec = open(pathS + "/book_Snonmem_recommendation", 'r')  # recommend for shadow
for line in Snonmem_rec.readlines():
    line = line.split('\t')
    sessionID = line[0]
    itemID = line[1]
    recommend_Snonmem.setdefault(int(sessionID), []).append(int(itemID))

# read interactions
itm = open(pathS + "/book_Smember_train", 'r')  # interactions for target_member
itn = open(pathS + "/book_Snonmem_train", 'r')  # interactions for target_member
interaction_Smember = {}  # interactions for target_member
interaction_Snonmem = {}  # interactions for target_nonmem
for line in itm.readlines():
    line = line.split('\t')
    if line[0] != 'SessionID':
        sessionID = line[0]
        itemID = line[1]
        interaction_Smember.setdefault(int(sessionID), []).append(int(itemID))
# print("length of interactions_Smember.keys:", len(interaction_Smember.keys()))
# print("interactions_Smember.keys:", sorted(interaction_Smember.keys()))
# print(interaction_Smember)
# 1: [490, 490, 490], 2: [5, 5, 5], 3: [288, 5, 288, 
for line in itn.readlines():
    line = line.split(',')
    if line[0] != 'SessionID':
        sessionID = line[0]
        itemID = line[1]
        interaction_Snonmem.setdefault(int(sessionID), []).append(int(itemID))
# print(interaction_Snonmem)
# 2000: [489, 1010, 489, 1010], 2001: [5, 5], 2002: 
# vectorization for shadow_member
label_shadow_member = {}
vector_shadow_member = {}
vector_shadow_member1 = {}  # vectors for shadow member interaction
for key, value in interaction_Smember.items():
    # key是userID， value为推荐列表
    label_shadow_member[key] = torch.tensor([1])
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors1.keys():
            temp_vector = temp_vector + vectors1[value[i]]
        else:
            length = length - 1
    if length != 0:
        temp_vector = temp_vector / length
    vector_shadow_member1[key] = temp_vector
    # print("vector_shadow_member1[key]:", vector_shadow_member1[key] ) # 正常
vector_shadow_member2 = {}  # vectors for shadow member recommendation
for key, value in recommend_Smember.items():
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors1.keys():
            temp_vector = temp_vector + vectors1[value[i]]
        else:
            length = length - 1
    if length != 0:
        temp_vector = temp_vector / length
    vector_shadow_member2[key] = temp_vector
    # print("vector_shadow_member2[key]:", vector_shadow_member2[key]) # 正常
    vector_shadow_member[key] = vector_shadow_member1[key] - vector_shadow_member2[key]


# vectorization for shadow_nonmember
label_shadow_nonmem = {}
vector_shadow_nonmem = {}
vector_shadow_nonmem1 = {}  # vectors for shadow nonmember interaction
for key, value in interaction_Snonmem.items():
    # key是userID， value为交互历史
    label_shadow_nonmem[key] = torch.tensor([0])
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors1.keys():
            temp_vector = temp_vector + vectors1[value[i]]
        else:
            length = length - 1
    if length != 0:
        temp_vector = temp_vector / length
    vector_shadow_nonmem1[key] = temp_vector
    # print("vector_shadow_nonmem1[key]:", vector_shadow_nonmem1[key]) # 正常
vector_shadow_nonmem2 = {}  # vectors for shadow nonmember recommendation
for key, value in recommend_Snonmem.items():
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors1.keys():
            temp_vector = temp_vector + vectors1[value[i]]
        else:
            length = length - 1
    if length != 0:
        temp_vector = temp_vector / length
    vector_shadow_nonmem2[key] = temp_vector
    # print("Keys in vector_shadow_nonmem1:", vector_shadow_nonmem1.keys())
    # print("Keys in vector_shadow_nonmem2:", vector_shadow_nonmem2.keys())
    vector_shadow_nonmem[key] = vector_shadow_nonmem1[key] - vector_shadow_nonmem2[key]

num_shadow = len(vector_shadow_member)+len(vector_shadow_nonmem)  # 2069
vector_shadow = [[]]*num_shadow
label_shadow = [[]]*num_shadow
idx = -1
for key, value in vector_shadow_member.items():
    idx = idx + 1
    # print("value:", value)
    vector_shadow[idx] = value
    label_shadow[idx] = label_shadow_member[key].long()  # member
for key, value in vector_shadow_nonmem.items():
    idx = idx + 1
    vector_shadow[idx] = value
    label_shadow[idx] = label_shadow_nonmem[key].long()  # non_member



# ------------Target----------------
fr_vector_target = open(Vpath + "/book_itemMatrix.txt", 'r')
path1 = "/content/drive/MyDrive/MIA/Dataset/data/processed_book/"
itemDict1 = {}
with open(path1 + "book_itemDict", 'r') as f:
    for line1 in f.readlines():
        line1 = line1.strip().split('\t')
        itemDict1[int(line1[1])] = int(line1[0])
# print(itemDict1) # 正常
vectors2 = {}  # vectors for target items
index1 = -1
for line in fr_vector_target.readlines():
    index1 = index1 + 1
    line = line.split(' ')
    line = list(map(float, line))
    t_vectors = torch.tensor(line)
    # print(t_vectors) # 正常
    if index1 in itemDict1:
      vectors2[itemDict1[index1]] = t_vectors
      # print("yes") # 未输出
# print(vectors2) # 为空


# read recommends


# Tmember_rec = open(pathT + "/book_Tmember_recommendations", 'r') # no kg
Tmember_rec = open("/content/drive/MyDrive/Recommendation-system-based-on-ripplenet/book_Tmember_recommendations", 'r') 


recommend_Tmember = {}  
for line in Tmember_rec.readlines():
    line = line.split('\t')
    sessionID = line[0]
    itemID = line[1]
    recommend_Tmember.setdefault(int(sessionID), []).append(int(itemID))
recommend_Tnonmem = {}  # recommends for target_nonmem
Tnonmem_rec = open(pathT + "/book_Tnonmem_recommendation", 'r')  # recommend for target

for line in Tnonmem_rec.readlines():
    line = line.split('\t')
    sessionID = line[0]
    itemID = line[1]
    recommend_Tnonmem.setdefault(int(sessionID), []).append(int(itemID))

# read interactions
itm = open(pathT + "/book_Tmember_train", 'r')  # interactions for target_member
# itm = open(pathT + "/beauty_Tmember_train", 'r')  
# itm = open(pathT + "/ml-1m_Tmember_train", 'r')
interaction_Tmember = {}  # interactions for 成员
for line in itm.readlines():
    line = line.split('\t')
    if line[0] != 'SessionID':
        sessionID = line[0]
        itemID = line[1]
        interaction_Tmember.setdefault(int(sessionID), []).append(int(itemID))
# print(interaction_Tmember) # 互动正常
# 1: 2:
# itn = open(pathT + "/beauty_Tnonmem_train", 'r')  # interactions for target_nonmember
# itn = open(pathT + "/ml-1m_Tnonmem_train", 'r') 
itn = open(pathT + "/book_Tnonmem_train", 'r')  # interactions for target_nonmember

interaction_Tnonmem = {}  # interactions for 非成员
for line in itn.readlines():
    line = line.split(',')
    if line[0] != 'SessionID':
        sessionID = line[0]
        itemID = line[1]
        interaction_Tnonmem.setdefault(int(sessionID), []).append(int(itemID))
# print(interaction_Tnonmem) # 互动正常
# 1992: [533, 533], 1993: [5, 5], 1994: [592, 592]
# vectorization for target_member
label_target_member = {}
vector_target_member = {}
vector_target_member1 = {}
for key, value in interaction_Tmember.items():
    # key: userID, value: recommendation list
    label_target_member[key] = torch.tensor([1])
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    # print("length", length)
    for i in range(len(value)):
        if value[i] in vectors2.keys():
            temp_vector = temp_vector + vectors2[value[i]]
        else:
            length = length - 1
    if length != 0:
        temp_vector = temp_vector / length
    vector_target_member1[key] = temp_vector
    # print("vector_target_member1[key]",vector_target_member1[key]) # 0
vector_target_member2 = {}
for key, value in recommend_Tmember.items():
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors2.keys():
            temp_vector = temp_vector + vectors2[value[i]]
        else:
            length = length - 1
    if length != 0:
        temp_vector = temp_vector / length
    vector_target_member2[key] = temp_vector
    vector_target_member[key] = vector_target_member1[key] - vector_target_member2[key]
    # print("vector_target_member[key]:", vector_target_member[key]) # 0
# vectorization for target_nonmember
label_target_nonmem = {}
vector_target_nonmem = {}
vector_target_nonmem1 = {}  # vectors for shadow member interaction
for key, value in interaction_Tnonmem.items():
    # key: userID, value: recommendation list
    label_target_nonmem[key] = torch.tensor([0])
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors2.keys():
            temp_vector = temp_vector + vectors2[value[i]]
        else:
            length = length - 1
    if length != 0:
        temp_vector = temp_vector / length
    vector_target_nonmem1[key] = temp_vector
vector_target_nonmem2 = {}  # vectors for shadow member recommendation
for key, value in recommend_Tnonmem.items():
    temp_vector = torch.zeros(num_latent)
    length = len(value)
    for i in range(len(value)):
        if value[i] in vectors2.keys():
            temp_vector = temp_vector + vectors2[value[i]]
        else:
            length = length - 1
    if length != 0:
        temp_vector = temp_vector / length
    vector_target_nonmem2[key] = temp_vector
    vector_target_nonmem[key] = vector_target_nonmem1[key] - vector_target_nonmem2[key]

num_target = len(vector_target_member)+len(vector_target_nonmem)  # 1939
# print("num_target(3025):", num_target)

vector_target = [[]] * num_target
label_target = [[]] * num_target
idx1 = -1
for key, value in vector_target_member.items():
    idx1 = idx1 + 1
    vector_target[idx1] = value
    label_target[idx1] = label_target_member[key].long()  # member
for key, value in vector_target_nonmem.items():
    idx1 = idx1 + 1
    vector_target[idx1] = value
    label_target[idx1] = label_target_nonmem[key].long()  # non_member






DataDir = "../AttackData/"
if not os.path.exists(DataDir):
    os.mkdir(DataDir)
ShadowDataset = open("../AttackData/trainForClassifier_ACMC.txt", 'w')
TargetDataset = open("../AttackData/testForClassifier_ACMC.txt", 'w')
for i in range(num_shadow):
    # print("vector_shadow[i]:", vector_shadow[i])
    ShadowDataset.write(str(vector_shadow[i].unsqueeze(0).tolist()) + '\t' + str(label_shadow[i].tolist()) + '\n')
for i in range(num_target):
    # print("vector_target[i]:", vector_target[i])
    TargetDataset.write(str(vector_target[i].unsqueeze(0).tolist()) + '\t' + str(label_target[i].tolist()) + '\n')


set_seed(2021)
state = np.random.get_state()
np.random.shuffle(vector_shadow)
np.random.set_state(state)
np.random.shuffle(label_shadow)

mlp = MLP()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01, momentum=0.7)


# accuarcy
def AccuarcyCompute(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    #    print(pred.shape(),label.shape())
    test_np = (np.argmax(pred, 1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)


losses = []
acces = []
eval_losses = []
eval_acces = []

for x in range(20):
    train_loss = 0
    train_acc = 0
    for i in range(num_shadow):
        optimizer.zero_grad()

        inputs = torch.autograd.Variable(vector_shadow[i])
        labels = torch.autograd.Variable(label_shadow[i])

        outputs = mlp(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()

        _, pred = outputs.max(1)
        if(int(pred)) == labels.numpy()[0]:
            train_acc += 1

        # print(x, ":", AccuarcyCompute(outputs, labels))
    losses.append(train_loss / num_shadow)
    acces.append(train_acc / num_shadow)
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}'.format(x, train_loss / (num_shadow),
                                                                    train_acc / (num_shadow)))

acc_ans = 0
TruePositive = 0
FalsePositive = 0
TrueNegative = 0
FalseNegative = 0
for i in range(num_target):
    inputs = torch.autograd.Variable(vector_target[i])
    labels = torch.autograd.Variable(label_target[i])
    outputs = mlp(inputs)
    # print(outputs)
    _, pred = outputs.max(1)
    if int(pred) == labels.numpy()[0]: # 预测值pred==真实标签labels，说明模型预测正确
        acc_ans += 1
        if int(pred) == 1: 
            TruePositive = TruePositive + 1 # 预测为成员
        else:
            TrueNegative = TrueNegative + 1 # 预测为非成员
    else: # 模型预测不正确
        if int(pred) == 1:
            FalsePositive = FalsePositive + 1
        else:
            FalseNegative = FalseNegative + 1

print('TruePositive:', TruePositive)
print('FalsePositive:', FalsePositive)
print('TrueNegative:', TrueNegative)
print('FalseNegative', FalseNegative)
print("accuarcy: ")
print((acc_ans / num_target))
print("precsion: ")
print((TruePositive / (TruePositive + FalsePositive)))
print("recall: ")
print((TruePositive / (TruePositive + FalseNegative)))

TPRate = TruePositive / (TruePositive + FalseNegative)
FPRate = FalsePositive / (FalsePositive + TrueNegative)
area = 0.5 * TPRate * FPRate + 0.5 * (TPRate + 1) * (1 - FPRate)
print("AUC: ")
print(area)
