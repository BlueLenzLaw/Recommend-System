# coding: utf-8 -*-
"""
        代码5-8 LFM推荐系统——数据准备
"""
import pandas as pd
import pickle
import os
import random
import math
import json

class DataProcessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.getDF(self.file_path)
    
    def parse(self, path):
        g = open(path, 'rb')
        for l in g:
            yield json.loads(l)

    ## 转化为dataframe格式
    def getDF(self, path):
        i = 0
        df = {}
        for d in self.parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')
    
    def process(self):
        # print('开始转化用户数据（users.dat）...')
        # self.process_user_data()
        # print('开始转化电影数据（movies.dat）...')
        # self.process_movies_date()
        print('开始转化用户对电影评分数据（ratings.dat）...')
        self.process_rating_data()
        print('Over!')

    # def process_user_data(self, file='../data/ml-1m/users.dat'):
    #     if not os.path.exists("data/users.csv"):
    #         fp = pd.read_table(file, sep='::', engine='python',names=['userID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    #         fp.to_csv('data/users.csv', index=False)

    def process_rating_data(self):
        if not os.path.exists("基于近邻的推荐算法/data/ratings.csv"):
            fp = self.df[['reviewerID','asin', 'overall']]
            fp.to_csv('基于近邻的推荐算法/data/ratings.csv', index=False)

    # def process_movies_date(self, file='../data/ml-1m/movies.dat'):
    #     if not os.path.exists("data/movies.csv"):
    #         fp = pd.read_table(file, sep='::', engine='python',names=['MovieID', 'Title', 'Genres'])
    #         fp.to_csv('data/movies.csv', index=False)

    # 对用户进行有行为产品和无行为产品数据标记
    def get_pos_neg_item(self,file_path="基于近邻的推荐算法/data/ratings.csv"):
        if not os.path.exists("基于近邻的推荐算法/data/lfm_items.dict"):
            self.items_dict_path="基于近邻的推荐算法/data/lfm_items.dict"

            self.uiscores=pd.read_csv(file_path)
            self.user_ids=set(self.uiscores["reviewerID"].values)
            self.item_ids=set(self.uiscores["asin"].values)
            self.items_dict = {user_id: self.get_one(user_id) for user_id in list(self.user_ids)}

            fw = open(self.items_dict_path, 'wb')
            pickle.dump(self.items_dict, fw)
            fw.close()

    # 定义单个用户的正向和负向数据
    # 正向：用户有过评分的物品；负向：用户无评分的物品
    def get_one(self, user_id):
        print('为用户%s准备正向和负向数据...' % user_id)
        pos_item_ids = set(self.uiscores[self.uiscores['reviewerID'] == user_id]['asin'])
        # 对称差：x和y的并集减去交集
        neg_item_ids = self.item_ids ^ pos_item_ids
        neg_item_ids = list(neg_item_ids)[:len(pos_item_ids)]
        item_dict = {}
        for item in pos_item_ids: item_dict[item] = 1
        for item in neg_item_ids: item_dict[item] = 0
        return item_dict


if __name__ == '__main__':
    dp=DataProcessing('基于近邻的推荐算法/All_Beauty_5.json')
    dp.process()
    dp.get_pos_neg_item()
