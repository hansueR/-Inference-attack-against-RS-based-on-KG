import numpy as np
import pandas as pd
import random

import torch

numOfRecommend = 100


def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def recommend(filename, filepath):
    rec = open(filepath, 'w')
    # data = pd.read_csv(filename, sep='\t', names=['UserID', 'ItemID', 'Rating', 'Time'])
    # data = pd.read_csv(filename, sep=',', names=['UserID', 'ItemID', 'Rating', 'Time'])
    data = pd.read_csv(filename, sep=',', names=['UserID', 'ItemID', 'Rating'])
    Popularity = data.groupby('ItemID').size()
    sorted_Popularity = Popularity.sort_values(ascending=False)
    popular_item = sorted_Popularity.index.tolist()
    user_data = data.groupby('UserID')
    for user, value in user_data:
        interacted_item = value['ItemID'].tolist()
        index = 0 
        for j in range(numOfRecommend):
            # print("j", j)
            while popular_item[index] in interacted_item:
                index = index + 1
            rec.write(str(user) + '\t' + str(popular_item[index]) + '\t' + '0' + '\n')
            index = index + 1

        # for j in range(numOfRecommend):
        # # 检查popular_item列表是否为空
        #   if not popular_item:
        #    print("popular_item列表为空")
        #    break
        #   while index < len(popular_item) and popular_item[index] in interacted_item:
        #    index = index + 1
        #    # 检查index是否超出popular_item列表长度
        #   if index >= len(popular_item):
        #     print("index超出popular_item列表长度")
        #     break
        #   rec.write(str(user) + '\t' + str(popular_item[index]) + '\t' + '0' + '\n')
        #   index = index + 1


if __name__ == '__main__':
    # nonmem_train = "/content/drive/MyDrive/MIA/Recommender/caser_pytorch-master/datasets/ml-1m_Snonmem_train"
    # nonmem_rec = "/content/drive/MyDrive/MIA/Recommender/caser_pytorch-master/datasets/ml-1m_Snonmem_recommendation"
    # recommend(nonmem_train, nonmem_rec)
  
    
    # Tnonmem_train = "/content/drive/MyDrive/MIA/Recommender/caser_pytorch-master/datasets/ml-1m_Tnonmem_train"
    # Tnonmem_rec = "/content/drive/MyDrive/MIA/Recommender/caser_pytorch-master/datasets/ml-1m_Tnonmem_recommendation"
    # recommend(Tnonmem_train, Tnonmem_rec)

    nonmem_train = "/content/drive/MyDrive/MIA/Recommender/caser_pytorch-master/datasets/book_Snonmem_train"
    nonmem_rec = "/content/drive/MyDrive/MIA/Recommender/caser_pytorch-master/datasets/book_Snonmem_recommendation"
    recommend(nonmem_train, nonmem_rec)  

    Tnonmem_train = "/content/drive/MyDrive/MIA/Recommender/caser_pytorch-master/datasets/book_Tnonmem_train"
    Tnonmem_rec = "/content/drive/MyDrive/MIA/Recommender/caser_pytorch-master/datasets/book_Tnonmem_recommendation"
    recommend(Tnonmem_train, Tnonmem_rec) 

    # nonmem_train = "/content/drive/MyDrive/MIA/Recommender/caser_pytorch-master/datasets/beauty_Snonmem_train"
    # nonmem_rec = "/content/drive/MyDrive/MIA/Recommender/caser_pytorch-master/datasets/beauty_Snonmem_recommendation"
    # recommend(nonmem_train, nonmem_rec)

    # Tnonmem_train = "/content/drive/MyDrive/MIA/Recommender/caser_pytorch-master/datasets/beauty_Tnonmem_train"
    # Tnonmem_rec = "/content/drive/MyDrive/MIA/Recommender/caser_pytorch-master/datasets/beauty_Tnonmem_recommendation"
    # recommend(Tnonmem_train, Tnonmem_rec)
    print('done')
