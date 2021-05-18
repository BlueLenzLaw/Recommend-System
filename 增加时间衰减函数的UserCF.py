import random
import math
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class NewUserCF:
    # 初始化函数,max_data 表示数据集中评分时间的最大值，即初始化时间衰减函数中的 t0
    def __init__(self,file_path):
        # 导入数据
        self.file_path = file_path
        self.df = self.getDF(self.file_path)
        self.k = 25
        self.nitems = 10
        self.alpha = 0.5
        self.beta = 0.8
        # 数据集分开
        self.train, self.test, self.max_data  = self.loadData()
        # 矩阵储存
        self.users_sim = self.UserSimilarityBest()

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
    
    # 加载数据集，并拆分成训练集和测试集
    def loadData(self):
        print("Start load Data and Split data ...")
        data = list()
        max_data = 0
        for i in range(self.df.shape[0]):
            userid,itemid,record, timestamp = self.df['reviewerID'][i], self.df['asin'][i], self.df['overall'][i], self.df['unixReviewTime'][i]
            data.append((userid, itemid, int(record), int(timestamp)))
            if int(timestamp) > max_data:
                max_data = int(timestamp)
        # 调用sklearn中的train_test_split拆分训练集和测试集
        train_list, test_list = train_test_split(data, test_size=0.1, random_state=40)
        # 将train 和 test 转化为字典格式方便调用
        train_dict = self.transform(train_list)
        test_dict = self.transform(test_list)
        return train_dict, test_dict, max_data

    # 将list转化为dict
    def transform(self, data):
        data_dict = dict()
        for user, item, record, timestamp in data:
            data_dict.setdefault(user, {}).setdefault(item, {})
            data_dict[user][item]["rate"] = record
            data_dict[user][item]["time"] = timestamp
        return data_dict
    
    # 计算用户之间的相似度，采用惩罚热门商品和优化算法复杂度的算法
    def UserSimilarityBest(self):
        print("Start calculation user's similarity...")
        if os.path.exists("基于近邻的推荐算法/user_sim_withtime.json"):
            print("从文件加载 ...")
            userSim = json.load(open("基于近邻的推荐算法/user_sim_withtime.json", "r"))
        else:
            # 得到每个item被哪些user评价过
            item_eval_by_users = dict()
            for u, items in self.train.items():
                for i in items.keys():
                    item_eval_by_users.setdefault(i, set())
                    if self.train[u][i]['rate'] > 0:
                        item_eval_by_users[i].add(u)
            # 构建倒排表
            count = dict()
            # 用户评价过多少个sku
            user_eval_item_count = dict()
            for i, users in item_eval_by_users.items():
                for u in users:
                    user_eval_item_count.setdefault(u, 0)
                    user_eval_item_count[u] += 1
                    count.setdefault(u, {})
                    for v in users:
                        count[u].setdefault(v, 0)
                        if u == v:
                            continue
                        count[u][v] += 1 / ( 1+ self.alpha * abs(self.train[u][i]["time"]-self.train[v][i]["time"]) / (24*60*60) ) \
                                           * 1 / math.log(1 + len(users))
            # 构建相似度矩阵
            userSim = dict()
            for u, related_users in count.items():
                userSim.setdefault(u, {})
                for v, cuv in related_users.items():
                    if u == v:
                        continue
                    userSim[u].setdefault(v, 0.0)
                    userSim[u][v] = cuv / math.sqrt(user_eval_item_count[u] * user_eval_item_count[v])
            json.dump(userSim, open('基于近邻的推荐算法/user_sim_withtime.json', 'w'))
        return userSim

    """
        为用户user进行物品推荐
            user: 为用户user进行推荐
            k: 选取k个近邻用户
            nitems: 取nitems个物品
    """
    def recommend(self, user):
        rank = dict()
        if (user in self.train.keys()):
            interacted_items = self.train.get(user, {})
            for v, wuv in sorted(self.users_sim[user].items(), key=lambda x: x[1], reverse=True)[0:self.k]:
                for i, rv in self.train[v].items():
                    if i in interacted_items:
                        continue
                    rank.setdefault(i, 0)
                    # rank[i] += wuv * rv["rate"]
                    rank[i] += wuv * rv["rate"] * 1/(1+ self.beta * ( self.max_data - abs(rv["time"]) ) )
            return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:self.nitems])
        else: 
            return rank
 
    """
        计算准确率
            k: 近邻用户数
            nitems: 推荐的item个数
    """
    def precision_(self):
        print("开始计算准确率 ...")
        hit = 0
        precision_ = 0
        for user in self.test.keys():
            tu = self.test.get(user, {}) ## 用户在测试集上实际喜欢的物品
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
        for user in self.test.keys():
            tu = self.test.get(user, {}) ## 用户在测试集上实际喜欢的物品 T(u)
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
        for user in self.test.keys():
            tu = self.test.get(user, {}) ## 用户在测试集上实际喜欢的物品 T(u)
            rank = self.recommend(user) # R(u)
            for item, rate in rank.items():
                if item in tu:
                    hit += 1
            precision += len(tu)
        return hit / precision

    def hit_rate(self):    
        print("开始计算命中率 ...")
        hit = 0
        for user in self.test.keys():
            tu = self.test.get(user, {}) ## 用户在测试集上实际喜欢的物品 T(u)
            rank = self.recommend(user) # R(u)
            cal = 0
            for item, rate in rank.items():
                if item in tu:
                    cal += 1
            if (cal != 0):
                hit += 1
        precision = len(self.test)
        return hit / precision

    def coverage(self):    
        print("开始计算覆盖率 ...")
        rec = []
        item_list = []
        for user in self.test.keys():
            tu = self.test.get(user, {}) ## 用户在测试集上实际喜欢的物品 T(u)
            for i in tu.keys():
                item_list.append(i)
            rank = self.recommend(user) # R(u)
            for item, rate in rank.items():
                if item in tu:
                    rec.append(item)
        return len(set(rec)) / len(set(item_list))

if __name__=='__main__':
    ncf = NewUserCF("基于近邻的推荐算法/All_Beauty_5.json")
    # print(ncf.test)
    # print(cf.loadData())
    # print(cf.trainData)
    # print(cf.trainData.keys())
    # result = cf.recommend("A3CIUOJXQ5VDQ2")
    # print("user '1' recommend result is {} ".format(result))

    precision_ = ncf.precision_()
    print("自带准确率为 {}%".format(round(100*precision_, 3)))

    precision = ncf.precision()
    print("precision is {}\%".format(round(100*precision,3)))

    recall = ncf.recall()
    print("recall is {}\%".format(round(100*recall,3)))

    hit_rate = ncf.hit_rate()
    print("hit_rate is {}\%".format(round(100*hit_rate,3)))

    coverage = ncf.coverage()
    print("coverage is {}\%".format(round(100*coverage,3)))
 