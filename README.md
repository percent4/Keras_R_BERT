本项目使用Keras实现R-BERT，在人物关系数据集上进行测试验证。

## 数据集

- 共3901条标注样本，训练集：测试集=8:2
- 标注样本：`亲戚  1837年6月20日，<e1>威廉四世</e1>辞世，他的侄女<e2>维多利亚</e2>即位。`，其中`亲戚`为关系，`威廉四世`为实体1（entity_1），`维多利亚`为实体2（entity_2）。
- 每一种关系的标注数量如下图:

<p float="left" align="center">
    <img width="600" src="https://raw.githubusercontent.com/percent4/R-BERT_for_people_relation_extraction/master/data/bar_chart.png" />  
</p>

## 模型结构

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68673458-1b090d00-0597-11ea-96b1-7c1453e6edbb.png" />  
</p>

## 运行环境

- python >= 3.6
- 第三方模块参考: `requiments.txt`

## 模型训练

```bash
$ python3 model_train.py
```

## 模型评估

```bash
$ python3 model_evalaute.py
# Model: chinese_L-12_H-768_A-12, weighted avgage F1 = 81.70%
# Model: chinese-roberta-wwm-ext, weighted avgage F1 = %
```

Model: chinese_L-12_H-768_A-12, 详细的评估结果如下：

```
              precision    recall  f1-score   support

     unknown     0.7980    0.7751    0.7864       209
         上下级     0.6286    0.7097    0.6667        31
          亲戚     0.8333    0.6250    0.7143        24
        兄弟姐妹     0.7750    0.9118    0.8378        34
          合作     0.8776    0.7288    0.7963        59
          同人     0.9459    0.8974    0.9211        39
          同学     0.8261    0.7917    0.8085        24
          同门     0.9200    0.8846    0.9020        26
          夫妻     0.8514    0.7975    0.8235        79
          好友     0.7647    0.8667    0.8125        30
          师生     0.7576    0.6757    0.7143        37
          情侣     0.7714    0.8710    0.8182        31
          父母     0.8367    0.9609    0.8945       128
          祖孙     0.9130    0.8400    0.8750        25

    accuracy                         0.8183       776
   macro avg     0.8214    0.8097    0.8122       776
weighted avg     0.8210    0.8183    0.8170       776
```

## 模型预测

```bash
$ python3 model_predict.py
```


## References

- [NLP-progress Relation Extraction](http://nlpprogress.com/english/relationship_extraction.html)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [https://github.com/wang-h/bert-relation-classification](https://github.com/wang-h/bert-relation-classification)
- [R-BERT](https://github.com/monologg/R-BERT)
- [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/pdf/1905.08284.pdf)