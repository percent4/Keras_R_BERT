# -*- coding: utf-8 -*-
# @Time : 2021/3/24 20:50
# @Author : Jclian91
# @File : model_evaluate.py
# @Place : Yangpu, Shanghai
# 模型评估脚本
import os
import json
import pandas as pd
from sklearn.metrics import classification_report

from model_predict import predict_single_sample
from config import *


# 加载训练好的模型
with open(os.path.join(DATA_DIR, "label.json"), "r", encoding="utf-8") as f:
    label_dict = json.loads(f.read())


# 模型评估
def evaluate():
    test_df = pd.read_csv(TEST_FILE_PATH, sep="\t", header=None)
    true_y_list, pred_y_list = [], []
    for i in range(test_df.shape[0]):
        print("predict %d samples" % (i+1))
        true_y, content = test_df.iloc[i, :]
        pred_y = predict_single_sample(content)
        true_y_list.append(true_y)
        pred_y_list.append(pred_y)

    return classification_report(true_y_list, pred_y_list, digits=4)


if __name__ == '__main__':
    output_data = evaluate()
    print("model evaluate result:\n")
    print(output_data)