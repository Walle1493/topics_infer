import jieba
import pandas as pd

def stopwords_list(stopwords_path):
    """创建停用词列表"""
    stopwords = []
    with open(stopwords_path, encoding="utf-8") as f:
        for line in f.readlines():
            stopwords.append(line.strip())
    return stopwords


def tokenize(sentence, stopwords):
    """对句子进行中文分词"""
    sentence_depart = jieba.cut(sentence.strip())
    # stopwords = stopwords_list()
    out_str = ''
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
            out_str += word
            out_str += " "
    return out_str

def find_test_max(topics_list):
    """xxx"""
    max_list = []
    for topics in topics_list:
        max_index, max_value = -1, 0
        for index, value in topics:
            if value > max_value:
                max_index, max_value = index, value
        max_list.append(max_index)
    return max_list

def find_predict_max(topics_list):
    topics_list = list(topics_list)
    max_topic = topics_list.index(max(topics_list))
    return max_topic

def save_topics(top_topics, output_path):
    """训练：存储主题类别"""
    topic_df = pd.DataFrame(top_topics)
    topic_df.to_csv(output_path, header=None, index=None)

def save_topics_infer(inference, infer_path):
    """训练：存储主题推断"""
    infers = []
    most_topic = []
    for values in inference:
        infer = []
        for index, value in enumerate(values):
            infer.append(value)
        infers.append(infer)
        most_topic.append(infer.index(max(infer)))
    infers_df = pd.DataFrame(infers)
    infers_df.to_csv(infer_path)
    return most_topic

def save_topics_test(topics_test, output_path):
    """测试"""
    topic_df = pd.DataFrame(topics_test)
    topic_df.to_csv(output_path, header=None, index=None)
    max_list = find_test_max(list(topics_test))
    return max_list

def save_topic_predict(arr, output_path):
    """预测"""
    topic_df = pd.DataFrame(arr[0])
    topic_df.to_csv(output_path, header=None)
    max_topic = find_predict_max(arr[0])
    return max_topic
