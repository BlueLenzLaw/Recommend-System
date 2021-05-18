import random
import math
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

class ItemCFRec:
    def __init__(self,file_path):
        # 原始数据路径文件
        self.file_path = file_path
        self.df = self.getDF(self.file_path)
        # 测试集与训练集的比例
        # self.ratio = ratio
        # self.data = self.loadData()

        self.k = 25
        self.nitems = 10

        self.trainData,self.testData = self.loadData()
        self.items_sim = self.ItemSimilarityBest()
    
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
    # def splitData(self,k,seed,M=9):
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

    # 计算物品之间的相似度
    def ItemSimilarityBest(self):
        print("开始计算物品之间的相似度")
        if os.path.exists("基于近邻的推荐算法/item_sim.json"):
            print("物品相似度从文件加载 ...")
            itemSim = json.load(open("基于近邻的推荐算法/item_sim.json", "r"))
        else:
            itemSim = dict()
            item_user_count = dict()  # 得到每个物品有多少用户产生过行为
            count = dict()  # 共现矩阵
            for user, item in self.trainData.items():
                # print("user is {}".format(user))
                for i in item.keys():
                    item_user_count.setdefault(i, 0)
                    if self.trainData[str(user)][i] > 0.0:
                        item_user_count[i] += 1
                    for j in item.keys():
                        count.setdefault(i, {}).setdefault(j, 0)
                        if self.trainData[str(user)][i] > 0.0 and self.trainData[str(user)][j] > 0.0 and i != j:
                            count[i][j] += 1
            # 共现矩阵 -> 相似度矩阵
            for i, related_items in count.items():
                itemSim.setdefault(i, dict())
                for j, cuv in related_items.items():
                    itemSim[i].setdefault(j, 0)
                    itemSim[i][j] = cuv / math.sqrt(item_user_count[i] * item_user_count[j])
        json.dump(itemSim, open('基于近邻的推荐算法/item_sim.json', 'w'))
        return itemSim

    """
        为用户进行推荐
            user: 用户
            k: k个临近物品
            nitem: 总共返回n个物品
    """
    def recommend(self, user):
        result = dict()
        u_items = self.trainData.get(user, {})
        for i, pi in u_items.items():
            for j, wj in sorted(self.items_sim[i].items(), key=lambda x: x[1], reverse=True)[0:self.k]:
                if j in u_items:
                    continue
                result.setdefault(j, 0)
                result[j] += pi * wj

        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[0:self.nitems])

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


if __name__ == "__main__":
    ib = ItemCFRec("基于近邻的推荐算法/All_Beauty_5.json")
    #print("用户1进行推荐的结果如下：{}".format(ib.recommend("1")))
    # print("准确率为： {}".format(ib.precision()))
    # print(ib.df.shape[0])

    precision_ = ib.precision_()
    print("自带准确率为 {}\%".format(round(100*precision_, 3)))

    precision = ib.precision()
    print("precision is {}\%".format(round(100*precision,3)))

    recall = ib.recall()
    print("recall is {}\%".format(round(100*recall,3)))

    hit_rate = ib.hit_rate()
    print("hit_rate is {}\%".format(round(100*hit_rate,3)))

    coverage = ib.coverage()
    print("coverage is {}\%".format(round(100*coverage,3)))
