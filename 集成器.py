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
from UserCF import UserCFRec

cf = UserCFRec("基于近邻的推荐算法/All_Beauty_5.json")
precision_ = cf.precision_()
print("自带准确率为 {}\%".format(round(100*precision_, 3)))

