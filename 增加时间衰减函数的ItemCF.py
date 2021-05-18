import random
import math
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split


class NewItemBasedCF:
    # 初始化函数,max_data 表示数据集中评分时间的最大值，即初始化时间衰减函数中的 t0
    def __init__(self,file_path):
        # 原始数据路径文件
        self.file_path = file_path
        self.df = self.getDF(self.file_path)

        self.k = 25
        self.nitems = 10

        self.alpha = 0.5
        self.beta = 0.8
        
        self.train, self.test, self.max_data = self.loadData()  
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

    # 加载数据集，并拆分成训练集和测试集
    def loadData(self):
        print("Start load Data and Split data ...")
        data = list()
        max_data = 0
        for i in range(self.df.shape[0]):
            userid,itemid,record, timestamp= self.df['reviewerID'][i], self.df['asin'][i], self.df['overall'][i], self.df['unixReviewTime'][i]
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

    # 计算物品之间的相似度
    def ItemSimilarityBest(self):
        print("开始计算物品之间的相似度")
        if os.path.exists("基于近邻的推荐算法/item_sim_withtime.json.json"):
            print("从文件加载 ...")
            itemSim = json.load(open("基于近邻的推荐算法/item_sim_withtime.json", "r"))
        else:
            itemSim = dict()
            item_eval_by_user_count = dict()  # 得到每个物品有多少用户产生过行为
            count = dict()  # 共现矩阵
            for user, items in self.train.items():
                # print("user is {}".format(user))
                for i in items.keys():
                    item_eval_by_user_count.setdefault(i, 0)
                    if self.train[str(user)][i]["rate"] > 0.0:
                        item_eval_by_user_count[i] += 1
                    for j in items.keys():
                        count.setdefault(i, {}).setdefault(j, 0)
                        if self.train[str(user)][i]["rate"] > 0.0 and self.train[str(user)][j]["rate"] > 0.0 and i != j:
                            count[i][j] += 1 * 1 / ( 1+ self.alpha * abs(self.train[user][i]["time"]-self.train[user][i]["time"]) / (24*60*60) )
            # 共现矩阵 -> 相似度矩阵
            for i, related_items in count.items():
                itemSim.setdefault(i, {})
                for j, num in related_items.items():
                    itemSim[i].setdefault(j, 0)
                    itemSim[i][j] = num / math.sqrt(item_eval_by_user_count[i] * item_eval_by_user_count[j])
        json.dump(itemSim, open('基于近邻的推荐算法/item_sim_withtime.json', 'w'))
        return itemSim

    """
        为用户进行推荐
            user: 用户
            k: k个临近物品
            nitem: 总共返回n个物品
    """  
    def recommend(self, user):
        result = dict()
        u_items = self.train.get(user, {})
        for i, rate_time in u_items.items():
            for j, wj in sorted(self.items_sim[i].items(), key=lambda x: x[1], reverse=True)[0:self.k]:
                if j in u_items:
                    continue
                result.setdefault(j, 0)
                # result[j] += rate_time["rate"] * wj
                result[j] += rate_time["rate"] * wj * 1/(1+ self.beta * ( self.max_data - abs(rate_time["time"]) ) )

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


if __name__ == "__main__":
    nib = NewItemBasedCF("基于近邻的推荐算法/All_Beauty_5.json")
    #print("用户1进行推荐的结果如下：{}".format(ib.recommend("1")))
    # print("准确率为： {}".format(ib.precision()))
    # print(ib.df.shape[0])
    # print(nib.test)

    precision_ = nib.precision_()
    print("自带准确率为 {}%".format(round(100*precision_, 3)))

    precision = nib.precision()
    print("precision is {}\%".format(round(100*precision,3)))

    recall = nib.recall()
    print("recall is {}\%".format(round(100*recall,3)))

    hit_rate = nib.hit_rate()
    print("hit_rate is {}\%".format(round(100*hit_rate,3)))

    coverage = nib.coverage()
    print("coverage is {}\%".format(round(100*coverage,3)))
