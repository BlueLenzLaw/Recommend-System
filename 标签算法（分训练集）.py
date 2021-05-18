# -*-coding:utf-8-*-
from rake_nltk import Rake
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import random
import math
import os
import json
from sklearn.model_selection import train_test_split

class RecBasedTag:
    # 由于从文件读取为字符串，统一格式为整数，方便后续计算
    def __init__(self, file_path):
        ## 获取信息
        self.nitems = 5

        self.file_path = file_path
        self.df = self.getDF(self.file_path)

        self.user_ids = self.df['reviewerID'].unique()
        self.itemall = self.df['asin'].unique()

        ## 进行关键词提取
        key_words = []
        for index, row in self.df.iterrows():
            r = Rake() 
            test_word = str(row['reviewText']) + str(row['summary'])
            r.extract_keywords_from_text(test_word)
            key_words_dict_scores = r.get_word_degrees()
            key_words.append(list(key_words_dict_scores.keys()))
        self.df['Key_words'] = key_words

        self.trainData,self.testData, self.uiscores = self.loadData()
        
        # 用户对商品的评分
        self.userRateDict = self.getUserRate()
        # print(self.userRateDict)
        # 商品与标签的相关度
        self.ItemTagsDict = self.getItemTags()
        # print(self.ItemTagsDict)
        # # 用户对每个标签打标的次数统计和每个标签被所有用户打标的次数统计
        self.userTagDict, self.tagUserDict = self.getUserTagNum()
        # print(self.tagUserDict)
        # # 用户最终对每个标签的喜好程度
        self.userTagPre = self.getUserTagPre()
        # print(self.userTagPre)

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
    # 获取用户对商品的评分信息

    # 加载评分数据到data
    def loadData(self):
        print("加载数据...")
        data=[]
        for i in range(self.df.shape[0]):
            userid,itemid,record,key_words= self.df['reviewerID'][i], self.df['asin'][i], self.df['overall'][i], self.df['Key_words'][i]
            data.append((userid,itemid,int(record),key_words))
        # 调用sklearn中的train_test_split拆分训练集和测试集
        train_list, test_list = train_test_split(data, test_size=0.1, random_state=40)
        uiscores = pd.DataFrame(train_list, columns=['reviewerID','asin', 'overall', 'Key_words'])
        # 将train 和 test 转化为字典格式方便调用
        train_dict = self.transform(train_list)
        test_dict = self.transform(test_list)
        return train_dict, test_dict, uiscores

    # 将list转化为dict
    def transform(self, data):
        data_dict = dict()
        for user, item, record, key_words in data:
            # print(key_words)
            data_dict.setdefault(user, {})
            data_dict[user][item]= record
            # data_dict[user][item]['Key_words'] = key_words
        return data_dict

    def getUserRate(self):
        userRateDict = dict()
        # fr = open(self.user_rate_file, "r", encoding="utf-8")
        for i in range(self.uiscores.shape[0]):
            userID,itemid,rate = self.uiscores['reviewerID'][i], self.uiscores['asin'][i], self.uiscores['overall'][i]
            userRateDict.setdefault(userID, {})
            # 对听歌次数进行适当比例的缩放，避免计算结果过大
            userRateDict[userID][itemid] = float(rate) / 100
        return userRateDict

    def getItemTags(self):
        ItemTagsDict = dict()
        for i in range(self.uiscores.shape[0]):
            itemID, tag= self.uiscores['asin'][i], self.uiscores['Key_words'][i]
            ItemTagsDict.setdefault(itemID, {})
            for tagID in tag:
                ItemTagsDict[itemID][tagID] = 1
        return ItemTagsDict

    # 获取每个用户打标的标签和每个标签被所有用户打标的次数
    def getUserTagNum(self):
        userTagDict = dict()
        tagUserDict = dict()
        for i in range(self.uiscores.shape[0]):
            userID,itemid,tag = self.uiscores['reviewerID'][i], self.uiscores['asin'][i], self.uiscores['Key_words'][i]
            # 统计每个标签被打标的次数
            for tagID in tag:
                if tagID in tagUserDict.keys():
                    tagUserDict[tagID] += 1
                else:
                    tagUserDict[tagID] = 1
            # 统计每个用户对每个标签的打标次数
                userTagDict.setdefault(userID, {})
                if tagID in userTagDict[userID].keys():
                    userTagDict[userID][tagID] += 1
                else:
                    userTagDict[userID][tagID] = 1
        return userTagDict, tagUserDict

        # 获取用户对标签的最终兴趣度
    def getUserTagPre(self):
        userTagPre = dict()
        userTagCount = dict()
        # Num 为用户打标总条数
        Num = self.uiscores.shape[0]
        for i in range(self.uiscores.shape[0]):
            userID,itemid,tag = self.uiscores['reviewerID'][i], self.uiscores['asin'][i], self.uiscores['Key_words'][i]
            userTagPre.setdefault(userID, {})
            userTagCount.setdefault(userID, {})
            rate_ui = (
                self.userRateDict[userID][itemid]
                if itemid in self.userRateDict[userID].keys()
                else 0
            )
            for tagID in tag:
                if tagID not in userTagPre[userID].keys():
                    userTagPre[userID][tagID] = (
                        rate_ui * self.ItemTagsDict[itemid][tagID]
                    )
                    userTagCount[userID][tagID] = 1
                else:
                    userTagPre[userID][tagID] += (
                        rate_ui * self.ItemTagsDict[itemid][tagID]
                    )
                    userTagCount[userID][tagID] += 1

        for userID in userTagPre.keys():
            for tagID in userTagPre[userID].keys():
                tf_ut = self.userTagDict[userID][tagID] / sum(
                    self.userTagDict[userID].values()
                )
                idf_ut = math.log(Num * 1.0 / (self.tagUserDict[tagID] + 1))
                userTagPre[userID][tagID] = (
                    userTagPre[userID][tagID]/userTagCount[userID][tagID] * tf_ut * idf_ut
                )
        return userTagPre

    # 对用户进行商品推荐
    def recommendForUser(self, user, K, flag=True):
        userItemPreDict = dict()
        # 得到用户没有打标过的商品
        for item in self.itemall:
            if item in self.ItemTagsDict.keys():
                # 计算用户对商品的喜好程度
                for tag in self.userTagPre[user].keys():
                    rate_ut = self.userTagPre[user][tag]
                    rel_it = (
                        0
                        if tag not in self.ItemTagsDict[item].keys()
                        else self.ItemTagsDict[item][tag]
                    )
                    if item in userItemPreDict.keys():
                        userItemPreDict[item] += rate_ut * rel_it
                    else:
                        userItemPreDict[item] = rate_ut * rel_it
        newuserItemPreDict = dict()
        if flag:
            # 对推荐结果进行过滤，过滤掉用户已经购买过的商品
            for item in userItemPreDict.keys():
                if item not in self.userRateDict[user].keys():
                    newuserItemPreDict[item] = userItemPreDict[item]
            return sorted(
                newuserItemPreDict.items(), key=lambda k: k[1], reverse=True
            )[:K]
        else:
            # 表示是用来进行效果评估
            return sorted(
                userItemPreDict.items(), key=lambda k: k[1], reverse=True
            )[:K]
    
    # 效果评估 重合度
    def evaluate(self, user):
        K = len(self.userRateDict[user])
        recResult = self.recommendForUser(user, K=K, flag=False)
        count = 0
        for (asin, pre) in recResult:
            if asin in self.userRateDict[user]:
                count += 1
        return count * 1.0 / K

    def recommend(self, user):
        userItemPreDict = dict()
        if (user in self.trainData.keys()):
        # 得到用户没有打标过的商品
            for item in self.itemall:
                if item in self.ItemTagsDict.keys():
                    # 计算用户对商品的喜好程度
                    for tag in self.userTagPre[user].keys():
                        rate_ut = self.userTagPre[user][tag]
                        rel_it = (
                            0
                            if tag not in self.ItemTagsDict[item].keys()
                            else self.ItemTagsDict[item][tag]
                        )
                        if item in userItemPreDict.keys():
                            userItemPreDict[item] += rate_ut * rel_it
                        else:
                            userItemPreDict[item] = rate_ut * rel_it
            newuserItemPreDict = dict()
            # 对推荐结果进行过滤，过滤掉用户已经购买过的商品
            for item in userItemPreDict.keys():
                if item not in self.userRateDict[user].keys():
                    newuserItemPreDict[item] = userItemPreDict[item]
            return dict(sorted(
                    newuserItemPreDict.items(), key=lambda k: k[1], reverse=True
                )[:self.nitems])
        else:
            return userItemPreDict


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
    rbt = RecBasedTag('基于近邻的推荐算法/All_Beauty_5.json')
    user = "AZJMUP77WBQZQ"
    # print(rbt.recommend(user))
    # print(rbt.userRateDict[user])
    # print(rbt.recommendForUser(user, K=20, flag=True))
    # print(rbt.evaluate(user))
    # print(rbt.user_ids)
    # print(rbt.userRateDict)
    precision_ = rbt.precision_()
    print("自带准确率为 {}%".format(round(100*precision_, 3)))

    precision = rbt.precision()
    print("precision is {}\%".format(round(100*precision,3)))

    recall = rbt.recall()
    print("recall is {}\%".format(round(100*recall,3)))

    hit_rate = rbt.hit_rate()
    print("hit_rate is {}\%".format(round(100*hit_rate,3)))

    coverage = rbt.coverage()
    print("coverage is {}\%".format(round(100*coverage,3)))
