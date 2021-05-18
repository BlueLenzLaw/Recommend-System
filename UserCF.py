import random
import math
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class UserCFRec:
    def __init__(self,file_path):
        self.file_path = file_path
        self.df = self.getDF(self.file_path)
        # self.data = self.loadData()
        # self.trainData,self.testData = self.splitData(3,47)  # 训练集与数据集

        self.k = 25
        self.nitems = 10

        self.trainData,self.testData = self.loadData()
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

    # 加载评分数据到data
    def loadData(self):
        print("加载数据...")
        data=[]
        for i in range(self.df.shape[0]):
            userid,itemid,record = self.df['reviewerID'][i], self.df['asin'][i], self.df['overall'][i]
            data.append((userid,itemid,int(record)))
        # 调用sklearn中的train_test_split拆分训练集和测试集
        train_list, test_list = train_test_split(data, test_size=0.1, random_state=40)
        # 将train 和 test 转化为字典格式方便调用
        train_dict = self.transform(train_list)
        test_dict = self.transform(test_list)
        return train_dict, test_dict

    # 将list转化为dict
    def transform(self, data):
        data_dict = dict()
        for user, item, record in data:
            data_dict.setdefault(user, {})
            data_dict[user][item]= record
            # data_dict[user][item]["time"] = timestamp
        return data_dict

    # """
    #     拆分数据集为训练集和测试集
    #         k: 参数
    #         seed: 生成随机数的种子
    #         M: 随机数上限
    # """
    # def splitData(self,k,seed,M=8):
    #     print("训练数据集与测试数据集切分...")
    #     train,test = {},{}
    #     random.seed(seed)
    #     for user,item,record in self.data:
    #         if random.randint(0,M) == k:
    #             test.setdefault(user,{})
    #             test[user][item] = record
    #         else:
    #             train.setdefault(user,{})
    #             train[user][item] = record
    #     return train,test

    # 计算用户之间的相似度，采用惩罚热门商品和优化算法复杂度的算法
    def UserSimilarityBest(self):
        print("开始计算用户之间的相似度 ...")
        if os.path.exists("基于近邻的推荐算法/user_sim.json"):
            print("用户相似度从文件加载 ...")
            userSim = json.load(open("基于近邻的推荐算法/user_sim.json","r"))
        else:
            # 得到每个item被哪些user评价过
            item_users = dict()
            for u, items in self.trainData.items():
                for i in items.keys():
                    item_users.setdefault(i,set())
                    if self.trainData[u][i] > 0:
                        item_users[i].add(u)
            # 即形成字典，表示出item被哪些user评价过
            # 构建倒排表
            count = dict() ## 得到用户与用户之间的关联
            user_item_count = dict() ## 得到用户出现的次数
            for i, users in item_users.items():
                for u in users:
                    user_item_count.setdefault(u,0) # 在已有该键对时直接在原键对上家
                    user_item_count[u] += 1
                    count.setdefault(u,{})
                    for v in users:
                        count[u].setdefault(v, 0)
                        if u == v:
                            continue
                        count[u][v] += 1 / math.log(1+len(users))
            # 构建相似度矩阵
            userSim = dict() ## 相似矩阵
            for u, related_users in count.items():
                userSim.setdefault(u,{})
                for v, cuv in related_users.items():
                    if u==v:
                        continue
                    userSim[u].setdefault(v, 0.0)
                    userSim[u][v] = cuv / math.sqrt(user_item_count[u] * user_item_count[v])
            json.dump(userSim, open('基于近邻的推荐算法/user_sim.json', 'w')) ## 将相似矩阵储存
        return userSim

    """
        为用户user进行物品推荐
            user: 为用户user进行推荐
            k: 选取k个近邻用户
            nitems: 取nitems个物品
    """
    def recommend(self, user):
        result = dict()
        if (user in self.trainData.keys()):
            have_score_items = self.trainData.get(user, {}) ## 提取有评分记录的物品
            for v, wuv in sorted(self.users_sim[user].items(), key=lambda x: x[1], reverse=True)[0:self.k]: ## 提取相似度最高的k个用户          
                for i, rvi in self.trainData[v].items():
                    if i in have_score_items:
                        continue
                    result.setdefault(i, 0)
                    result[i] += wuv * rvi ## 计算用户对该物品的评分
            return dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[0:self.nitems]) ## 取前nitems位作为推荐
        else:
            return result

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

if __name__=='__main__':
    cf = UserCFRec("基于近邻的推荐算法/All_Beauty_5.json")
    # print(cf.loadData())
    # print(cf.trainData)
    # print(cf.trainData.keys())
    # result = cf.recommend("A3CIUOJXQ5VDQ2")
    # print("user '1' recommend result is {} ".format(result))

    precision_ = cf.precision_()
    print("自带准确率为 {}\%".format(round(100*precision_, 3)))

    precision = cf.precision()
    print("precision is {}\%".format(round(100*precision,3)))

    recall = cf.recall()
    print("recall is {}\%".format(round(100*recall,3)))

    hit_rate = cf.hit_rate()
    print("hit_rate is {}\%".format(round(100*hit_rate,3)))

    coverage = cf.coverage()
    print("coverage is {}\%".format(round(100*coverage,3)))
 