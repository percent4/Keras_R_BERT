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

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 200)          0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, 200)          0                                            
__________________________________________________________________________________________________
model_2 (Model)                 multiple             101677056   input_1[0][0]                    
                                                                 input_2[0][0]                    
__________________________________________________________________________________________________
input_3 (InputLayer)            (None, 200)          0                                            
__________________________________________________________________________________________________
input_4 (InputLayer)            (None, 200)          0                                            
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 768)          0           model_2[1][0]                    
__________________________________________________________________________________________________
dot_1 (Dot)                     (None, 768)          0           input_3[0][0]                    
                                                                 model_2[1][0]                    
__________________________________________________________________________________________________
dot_2 (Dot)                     (None, 768)          0           input_4[0][0]                    
                                                                 model_2[1][0]                    
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 768)          0           lambda_1[0][0]                   
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 768)          0           dot_1[0][0]                      
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 768)          0           dot_2[0][0]                      
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 768)          590592      dropout_1[0][0]                  
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 768)          590592      dropout_2[0][0]                  
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 768)          590592      dropout_3[0][0]                  
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 2304)         0           dense_1[0][0]                    
                                                                 dense_2[0][0]                    
                                                                 dense_3[0][0]                    
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 14)           32270       concatenate_1[0][0]              
==================================================================================================
Total params: 103,481,102
Trainable params: 103,481,102
Non-trainable params: 0
__________________________________________________________________________________________________
```

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
# Model: chinese_L-12_H-768_A-12, weighted avgage F1 = 82.74%
# Model: chinese-RoBERTa-wwm-ext, weighted avgage F1 = 85.27%
```

Model: chinese_L-12_H-768_A-12, 详细的评估结果如下：

```
              precision    recall  f1-score   support

     unknown     0.8216    0.7273    0.7716       209
         上下级     0.6047    0.8387    0.7027        31
          亲戚     0.8889    0.6667    0.7619        24
        兄弟姐妹     0.7692    0.8824    0.8219        34
          合作     0.8276    0.8136    0.8205        59
          同人     1.0000    0.9231    0.9600        39
          同学     0.9524    0.8333    0.8889        24
          同门     1.0000    0.8846    0.9388        26
          夫妻     0.7816    0.8608    0.8193        79
          好友     0.8276    0.8000    0.8136        30
          师生     0.7561    0.8378    0.7949        37
          情侣     0.7941    0.8710    0.8308        31
          父母     0.8582    0.9453    0.8996       128
          祖孙     0.9524    0.8000    0.8696        25

    accuracy                         0.8273       776
   macro avg     0.8453    0.8346    0.8353       776
weighted avg     0.8344    0.8273    0.8274       776
```

Model: chinese-RoBERTa-wwm-ext, 详细的评估结果如下：

```
              precision    recall  f1-score   support

     unknown     0.7930    0.8612    0.8257       209
         上下级     0.7188    0.7419    0.7302        31
          亲戚     0.8824    0.6250    0.7317        24
        兄弟姐妹     0.8378    0.9118    0.8732        34
          合作     0.8600    0.7288    0.7890        59
          同人     1.0000    0.9487    0.9737        39
          同学     0.8800    0.9167    0.8980        24
          同门     0.9615    0.9615    0.9615        26
          夫妻     0.8333    0.8861    0.8589        79
          好友     0.8065    0.8333    0.8197        30
          师生     0.8857    0.8378    0.8611        37
          情侣     0.9231    0.7742    0.8421        31
          父母     0.9062    0.9062    0.9062       128
          祖孙     0.9524    0.8000    0.8696        25

    accuracy                         0.8531       776
   macro avg     0.8743    0.8381    0.8529       776
weighted avg     0.8566    0.8531    0.8527       776
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
- [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)