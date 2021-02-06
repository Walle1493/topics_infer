# 说明

使用gensim包的LDA模块进行主题分类推理建模

#### 数据集

> 从网页中爬取的新闻文本，数据来源可以联系作者
>
> 训练集：`377,86`个样本

#### 超参数

> `num_topics` 表示分类的主题数目 --`default`=10
>
> `num_words` 表示每个主题有几个关键词 --`default`=6

#### 文件说明

> 将所有数据放在`data`目录下
>
> 将输出文件放在`output`目录下
>
> `pick`存放训练结果的二进制文件

#### 运行

```sh
mkdir data
mkdir output
mkdir pick
pip install requirements.txt
sh run_lda.sh
```



#### 参考项目

`https://github.com/DengYangyong/LDA_gensim.git`

