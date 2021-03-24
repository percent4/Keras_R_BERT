# -*- coding: utf-8 -*-
# @Time : 2021/3/24 20:25
# @Author : Jclian91
# @File : check_data.py
# @Place : Yangpu, Shanghai
with open("train.csv", "r", encoding="utf-8") as f:
    content = [_.strip() for _ in f.readlines()]

for line in content:
    if "<e1>" in line and "</e1>" in line:
        pass
    else:
        print(line)