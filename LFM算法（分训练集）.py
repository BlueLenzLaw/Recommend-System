# coding: utf-8 -*-
import random
import pickle
import pandas as pd
import numpy as np
from math import exp
import time
from sklearn.model_selection import train_test_split
import os

class LFM:

    def __init__(self):
        self.nitems = 10
        self.class_count = 10
        self.iter_count = 50 
        self.lr = 0.02
        self.lam = 0.01
        self._init_model()

    """
        初始化参数
            randn: 从标准正态分布中返回n个值
            pd.DataFrame: columns 指定列顺序，index 指定索引

            self.uiscores
    """
    def _init_model(self):
        file_path = '基于近邻的推荐算法/data/ratings.csv'
        # pos_neg_path = '"基于近邻的推荐算法/data/lfm_items_train_version.dict"'

        ## 对此做训练集测试集以及self.uiscores的修改
        self.df = pd.read_csv(file_path)
        self.trainData,self.testData,self.uiscores = self.loadData()  
        # print(self.testData)
        self.user_ids = set(self.df['reviewerID'].values)  # 991
        #print(len(self.user_ids))
        
        self.item_ids = set(self.df['asin'].values) # 85
        #print(len(self.item_ids))
        self.items_dict = self.get_pos_neg_item()
        # self.items_dict = pickle.load(open(pos_neg_path,'rb'))
        # print(self.items_dict)
        
        array_p = np.random.randn(len(self.user_ids), self.class_count)
        array_q = np.random.randn(len(self.item_ids), self.class_count)
        self.p = pd.DataFrame(array_p, columns=range(0, self.class_count), index=list(self.user_ids))
        self.q = pd.DataFrame(array_q, columns=range(0, self.class_count), index=list(self.item_ids))
        # print(array_p, self.p)

    def loadData(self):
        print("加载数据...")
        data=[]
        for i in range(self.df.shape[0]):
            userid,itemid,record = self.df['reviewerID'][i], self.df['asin'][i], self.df['overall'][i]
            data.append((userid,itemid,int(record)))
        # 调用sklearn中的train_test_split拆分训练集和测试集
        train_list, test_list = train_test_split(data, test_size=0.1, random_state=40)
        uiscores = pd.DataFrame(train_list, columns=['reviewerID','asin', 'overall'])  ## 提取训练集的矩阵数据
        # 将train 和 test 转化为字典格式方便调用
        train_dict = self.transform(train_list)
        test_dict = self.transform(test_list)
        return train_dict, test_dict, uiscores
    
    # 将list转化为dict
    def transform(self, data):
        data_dict = dict()
        for user, item, record in data:
            data_dict.setdefault(user, {})
            data_dict[user][item]= record
            # data_dict[user][item]["time"] = timestamp
        return data_dict

        # 对用户进行有行为产品和无行为产品数据标记
    def get_pos_neg_item(self):
        items_dict_path="基于近邻的推荐算法/data/lfm_items_train_version.dict"
        if not os.path.exists(items_dict_path):
            items_dict = {user_id: self.get_one(user_id) for user_id in list(self.user_ids)}
            
            fw = open(items_dict_path, 'wb')
            pickle.dump(items_dict, fw)
            fw.close()
        else:
            items_dict = pickle.load(open(items_dict_path,'rb'))
        
        return items_dict


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


    """
        计算用户 user_id 对 item_id的兴趣度
            p: 用户对每个类别的兴趣度
            q: 物品属于每个类别的概率
    """
    def _predict(self, user_id, item_id):
        p = np.mat(self.p.loc[user_id].values)
        q = np.mat(self.q.loc[item_id].values).T
        r = (p * q).sum()
        # 借助sigmoid函数，转化为是否感兴趣
        logit = 1.0 / (1 + exp(-r))
        return logit

    # 使用误差平方和(SSE)作为损失函数
    def _loss(self, user_id, item_id, y, step):
        e = y - self._predict(user_id, item_id)
        # print('Step: {}, user_id: {}, item_id: {}, y: {}, loss: {}'.format(step, user_id, item_id, y, e))
        return e

    """
        使用随机梯度下降算法求解参数，同时使用L2正则化防止过拟合
        eg:
            E = 1/2 * (y - predict)^2, predict = matrix_p * matrix_q
            derivation(E, p) = -matrix_q*(y - predict), derivation(E, q) = -matrix_p*(y - predict),
            derivation（l2_square，p) = lam * p, derivation（l2_square, q) = lam * q
            delta_p = lr * (derivation(E, p) + derivation（l2_square，p))
            delta_q = lr * (derivation(E, q) + derivation（l2_square, q))
    """
    def _optimize(self, user_id, item_id, e):
        gradient_p = -e * self.q.loc[item_id].values
        l2_p = self.lam * self.p.loc[user_id].values
        delta_p = self.lr * (gradient_p + l2_p)

        gradient_q = -e * self.p.loc[user_id].values
        l2_q = self.lam * self.q.loc[item_id].values
        delta_q = self.lr * (gradient_q + l2_q)

        self.p.loc[user_id] -= delta_p
        self.q.loc[item_id] -= delta_q

    # 训练模型，每次迭代都要降低学习率，刚开始由于离最优值相差较远，因此下降较快，当到达一定程度后，就要减小学习率
    def train(self):
        for step in range(0, self.iter_count):
            time.sleep(30)
            for user_id, item_dict in self.items_dict.items():
                print('Step: {}, user_id: {}'.format(step, user_id))
                item_ids = list(item_dict.keys())
                random.shuffle(item_ids)
                for item_id in item_ids:
                    e = self._loss(user_id, item_id, item_dict[item_id], step)
                    self._optimize(user_id, item_id, e)
            self.lr *= 0.9
        self.save()

    # 计算用户未评分过的产品，并取top N返回给用户
    def predict(self, user_id, top_n=10):
        self.load()
        user_item_ids = set(self.uiscores[self.uiscores['reviewerID'] == user_id]['asin'])
        other_item_ids = self.item_ids ^ user_item_ids # 交集与并集的差集
        interest_list = [self._predict(user_id, item_id) for item_id in other_item_ids]
        candidates = sorted(zip(list(other_item_ids), interest_list), key=lambda x: x[1], reverse=True)
        return candidates[:top_n]

    # 保存模型
    def save(self):
        f = open('基于近邻的推荐算法/data/lfm.model', 'wb')
        pickle.dump((self.p, self.q), f)
        f.close()

    # 加载模型
    def load(self):
        f = open('基于近邻的推荐算法/data/lfm.model', 'rb')
        self.p, self.q = pickle.load(f)
        f.close()

    # 模型效果评估，从所有user中随机选取10个用户进行评估,评估方法为：绝对误差（AE）
    # def evaluate(self):
    #     self.load()
    #     users=random.sample(self.user_ids,10)
    #     user_dict={}
    #     for user in users:
    #         user_item_ids = set(self.uiscores[self.uiscores['reviewerID'] == user]['asin'])
    #         # print(user_item_ids)
    #         _sum=0.0
    #         for item_id in user_item_ids:
    #             p = np.mat(self.p.loc[user].values)
    #             q = np.mat(self.q.loc[item_id].values).T
    #             _r = (p * q).sum() 
    #             # print(p)
    #             # print(_r)
    #             r=self.uiscores[(self.uiscores['reviewerID'] == user)
    #                             & (self.uiscores['asin']==item_id)]["overall"].values[0]
    #             _sum+=abs(r-_r)
    #             #print(r)
    #         user_dict[user] = _sum/len(user_item_ids)
    #         print("userID：{},AE：{}".format(user,user_dict[user]))

    #     return sum(user_dict.values())/len(user_dict.keys())

    # 推荐取top N返回给用户计算推荐系统指标
    def recommend(self, user_id):
        self.load()
        user_item_ids = set(self.uiscores[self.uiscores['reviewerID'] == user_id]['asin'])
        other_item_ids = self.item_ids ^ user_item_ids # 交集与并集的差集
        interest_list = [self._predict(user_id, item_id) for item_id in self.item_ids]
        candidates = sorted(zip(list(self.item_ids), interest_list), key=lambda x: x[1], reverse=True)
        return dict(candidates[:self.nitems])

    """
        计算准确率
            k: 近邻用户数
            nitems: 推荐的item个数
    """
    def precision_(self):
        print("开始计算准确率 ...")
        hit = 0
        precision_ = 0
        for user in self.testData.keys():
            tu = self.testData.get(user, {}) ## 用户在测试集上实际喜欢的物品
            rank = self.recommend(user) # 推荐的物品
            for item, rate in rank.items():
                if item in tu:
                    hit += 1
            precision_ += self.nitems
        return hit / (precision_ * 1.0)

    def precision(self):
        print("开始计算精确率 ...")
        hit = 0
        precision = 0
        for user in self.testData.keys():
            tu = self.testData.get(user, {}) ## 用户在测试集上实际喜欢的物品 T(u)
            rank = self.recommend(user) # R(u)
            for item, rate in rank.items():
                if item in tu:
                    hit += 1
            precision += len(rank)
        return hit / precision

    def recall(self):    
        print("开始计算召回率 ...")
        hit = 0
        precision = 0
        for user in self.testData.keys():
            tu = self.testData.get(user, {}) ## 用户在测试集上实际喜欢的物品 T(u)
            rank = self.recommend(user) # R(u)
            for item, rate in rank.items():
                if item in tu:
                    hit += 1
            precision += len(tu)
        return hit / precision

    def hit_rate(self):    
        print("开始计算命中率 ...")
        hit = 0
        for user in self.testData.keys():
            tu = self.testData.get(user, {}) ## 用户在测试集上实际喜欢的物品 T(u)
            rank = self.recommend(user) # R(u)
            cal = 0
            for item, rate in rank.items():
                if item in tu:
                    cal += 1
            if (cal != 0):
                hit += 1
        precision = len(self.testData)
        return hit / precision

    def coverage(self):    
        print("开始计算覆盖率 ...")
        rec = []
        item_list = []
        for user in self.testData.keys():
            tu = self.testData.get(user, {}) ## 用户在测试集上实际喜欢的物品 T(u)
            for i in tu.keys():
                item_list.append(i)
            rank = self.recommend(user) # R(u)
            for item, rate in rank.items():
                if item in tu:
                    rec.append(item)
        return len(set(rec)) / len(set(item_list))

if __name__=="__main__":
    lfm=LFM()

    # print(lfm.trainData)
    # print(lfm.items_dict['A344ILJPHYQ0V']['B00006L9LC'])#['B000URXP6E']
    lfm.train()
    
    # print(lfm.predict('A265U4400IMZN4',10))
    # print(lfm.recommend('A265U4400IMZN4',10))
    # # print(lfm.evaluate())
    precision_ = lfm.precision_()
    print("自带准确率为 {}%".format(round(100*precision_, 3)))

    precision = lfm.precision()
    print("precision is {}\%".format(round(100*precision,3)))

    recall = lfm.recall()
    print("recall is {}\%".format(round(100*recall,3)))

    hit_rate = lfm.hit_rate()
    print("hit_rate is {}\%".format(round(100*hit_rate,3)))

    coverage = lfm.coverage()
    print("coverage is {}\%".format(round(100*coverage,3)))
 
    