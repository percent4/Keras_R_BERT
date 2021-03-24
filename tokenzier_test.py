# -*- coding: utf-8 -*-
# @Time : 2021/3/24 18:46
# @Author : Jclian91
# @File : tokenzier_test.py
# @Place : Yangpu, Shanghai
import codecs
from keras_bert import Tokenizer

# 建议长度<=510
maxlen = 128
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'


token_dict = {}
with codecs.open(dict_path, 'r', 'utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            else:
                R.append('[UNK]')   # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)

text = "$胡果$(1919～1979年)，原名#胡国梁#，祖籍南京。"
entity = "胡果"

x1, x2 = tokenizer.encode(first=text, max_len=maxlen)
x1, x2 = x1[:-1], x2[:-1]
print(x1, x2)

x1, x2 = tokenizer.encode(first=entity, max_len=maxlen)
x1, x2 = x1[:-1], x2[:-1]
print(x1, x2)