import jieba 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import json


def jieba_tokenize(text):
    """分词"""
    return jieba.lcut(text) 


def load_data(file_path):
    with open(file_path, encoding="utf-8") as f:
        data = f.readlines()
    data = [item.strip() for item in data]
    return data


def store_cluster(data, label, k):
    d = {}
    ls = []
    for i in range(len(data)):
        if label[i] in d:
            d[label[i]].append(data[i])
        else:
            d[label[i]] = [data[i]]

    for i in range(k):
        curr = {}
        curr["number"] = i
        curr["topics"] = d[i]
        curr["class"] = d[i][0]
        ls.append(curr)
    
    with open("word_output/first_cluster.json", "w", encoding="gbk") as f:
        json.dump(ls, f, indent=2, ensure_ascii=False)


def main():
    tfidf_vectorizer = TfidfVectorizer(tokenizer=jieba_tokenize,lowercase=False)
    '''
    tokenizer: 指定分词函数
    lowercase: 在分词之前将所有的文本转换成小写，因为涉及到中文文本处理，
    所以最好是False
    '''
    
    text_list = load_data("data/topics.txt")
    
    #需要进行聚类的文本集
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_list)
    
    # 超参：类别数
    num_clusters = 8
    km_cluster = KMeans(n_clusters=num_clusters, max_iter=800, n_init=40, 
                        init='k-means++',n_jobs=-1)
    '''
    n_clusters: 指定K的值
    max_iter: 对于单次初始值计算的最大迭代次数
    n_init: 重新选择初始值的次数
    init: 制定初始值选择的算法
    n_jobs: 进程个数，为-1的时候是指默认跑满CPU
    注意，这个对于单个初始值的计算始终只会使用单进程计算，
    并行计算只是针对与不同初始值的计算。比如n_init=10，n_jobs=40, 
    服务器上面有20个CPU可以开40个进程，最终只会开10个进程
    '''
    
    
    #返回各自文本的所被分配到的类索引
    result = km_cluster.fit_predict(tfidf_matrix)
    
    print("Predicting result:", result)
    # print(len(result))    # 335
    '''
    每一次fit都是对数据进行拟合操作，
    所以我们可以直接选择将拟合结果持久化，
    然后预测的时候直接加载，进而节省时间。
    '''
    store_cluster(text_list, result, num_clusters)
    
    
    # 用joblib存储pkl
    # joblib.dump(tfidf_vectorizer, 'word_output/tfidf_fit_result.pkl')
    # joblib.dump(km_cluster, 'word_output/km_cluster_fit_result.pkl')
    
    #程序下一次则可以直接load
    # tfidf_vectorizer = joblib.load('word_output/tfidf_fit_result.pkl')
    # km_cluster = joblib.load('word_output/km_cluster_fit_result.pkl')


def second_cluster():
    tfidf_vectorizer = TfidfVectorizer(tokenizer=jieba_tokenize,lowercase=False)

    # load data
    with open("word_output/first_cluster.json", encoding="utf-8") as f:
        complete_data = json.load(f)
    data = complete_data[0]["topics"]

    tfidf_matrix = tfidf_vectorizer.fit_transform(data)

    # 超参：类别数
    num_clusters = 8
    km_cluster = KMeans(n_clusters=num_clusters, max_iter=600, n_init=80, 
                        init='k-means++',n_jobs=-1)
    
    label = km_cluster.fit_predict(tfidf_matrix)
    
    # store cluster
    with open("word_output/first_cluster.json", encoding="utf-8") as f:
        cluster = json.load(f)
    d = {}
    ls = []
    # construct dictionary
    for i in range(len(data)):
        if label[i] in d:
            d[label[i]].append(data[i])
        else:
            d[label[i]] = [data[i]]
    # construct list
    for i in range(num_clusters):
        curr = {}
        curr["number"] = i
        curr["topics"] = d[i]
        curr["class"] = d[i][0]
        ls.append(curr)
    cluster[0]["topics"] = ls
    with open("word_output/second_cluster.json", "w", encoding="utf-8") as f:
        json.dump(cluster, f, indent=2, ensure_ascii=False)


def third_cluster():
    tfidf_vectorizer = TfidfVectorizer(tokenizer=jieba_tokenize,lowercase=False)

    # load data
    with open("word_output/second_cluster.json", encoding="utf-8") as f:
        complete_data = json.load(f)
    # recluster data0 and data3
    data = list((complete_data[0]["topics"][0]["topics"], complete_data[0]["topics"][3]["topics"]))
    num_clusters = [8, 64]
    label = []
    for index, data_detail in enumerate(data):
        tfidf_matrix = tfidf_vectorizer.fit_transform(data_detail)
        km_cluster = KMeans(n_clusters=num_clusters[index], max_iter=400, n_init=40, init='k-means++',n_jobs=-1)
        label_detail = km_cluster.fit_predict(tfidf_matrix)
        label.append(label_detail)
    # print(data)
    # print(label)
    max_label = []
    for label_detail in label:
        curr_max = max(list(label_detail), key=list(label_detail).count)
        max_label.append(curr_max)
    # print(max_label)
    # store cluster
    with open("word_output/second_cluster.json", encoding="utf-8") as f:
        cluster = json.load(f)
    for i in range(len(data)):
        d = {}
        ls = []
        # construct dictionary
        for j in range(len(data[i])):
            if label[i][j] == max_label[i]:
                this_label = 0
            else:
                this_label = 1
            if this_label in d:
                d[this_label].append(data[i][j])
            else:
                d[this_label] = [data[i][j]]
        # print(d)
        # construct list
        for j in range(len(num_clusters)):
            curr = {}
            curr["number"] = j
            curr["topics"] = d[j]
            curr["class"] = d[j][0]
            ls.append(curr)
        # print(ls)
        if i == 0:
            cluster[0]["topics"][0]["topics"] = ls
        else:
            cluster[0]["topics"][3]["topics"] = ls
    with open("word_output/third_cluster.json", "w", encoding="utf-8") as f:
        json.dump(cluster, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # main()
    # second_cluster()
    third_cluster()
