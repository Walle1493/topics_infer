import jieba
import os
import re
import argparse
import pickle
import pandas as pd
from gensim import corpora, models, similarities
from utils import *
import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# * Load input data.
def load_documents(input_path):
    """Load input data"""
    with open(input_path, encoding="utf-8") as f:
        docs = f.readlines()
    return docs

# * Pre-process that data.
def pre_process(docs, stopwords_path):
    """Pre-process that data"""
    stopwords = stopwords_list(stopwords_path)

    new_docs = []
    # 给每个文档分词
    for doc in docs:
        doc = re.sub(r'[^\u4e00-\u9fa5]+','', doc)
        doc_token = tokenize(doc.strip(), stopwords)
        new_docs.append(doc_token)
    
    train_docs = []
    for doc in new_docs:
        train_doc = [word.strip() for word in doc.split()]
        train_docs.append(train_doc)

    return train_docs

# * Transform documents into bag-of-words vectors.
# * Construct dictionary.
def construct_dictionary(docs):
    """Construct word frequency dictionary"""
    dictionary = corpora.Dictionary(docs)
    return dictionary

# * Train an LDA model.
def train(dictionary, docs, output_path, infer_path, num_topics=10, num_words=10):
    """Train an LDA model"""
    # dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(text) for text in docs]
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    top_topics = lda.print_topics(num_words=num_words)
    save_topics(top_topics, output_path)
    # 预测每个文章最可能的分类
    most_topic = save_topics_infer(lda.inference(corpus)[0], infer_path)
    # TO-DO
    # print(most_topic)

    print("Train finished")
    # for topic in top_topics:
    #     print(topic)
    return lda

def test(lda, dictionary, test_docs, test_output):
    # 新闻ID化    
    corpus_test = [dictionary.doc2bow(text) for text in test_docs]
    topics_test = lda.get_document_topics(corpus_test)
    max_list = save_topics_test(topics_test, test_output)
    print("Most possible topics:")
    print(max_list)
    print("Test finished")
    # print(list(topics_test))
    # labels = ['体育','娱乐','科技']
    # for i in range(3):
    #     print('这条'+labels[i]+'新闻的主题分布为：\n')
    #     print(topics_test[i],'\n')

def predict(lda, dictionary, pred_doc, pred_output):
    bow = dictionary.doc2bow(pred_doc)
    arr = lda.inference([bow])[0]
    max_topic = save_topic_predict(arr, pred_output)
    print("Most possible topic:")
    print(max_topic)
    print("Predict finished")

# def use_later():
#     print(texts[e])
#     for ee, value in enumerate(values):
#         print('\t主题%d推断值%.2f' % (ee, value))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stopwords_path", type=str, default="data/stopwords.txt")
    parser.add_argument("--input_path", type=str, default="data/original_corpus_minus.txt")
    parser.add_argument("--test_path", type=str, default="data/test_set.txt")
    parser.add_argument("--predict_path", type=str, default="data/predict_document.txt")
    parser.add_argument("--topics_path", type=str, default="output/top_topics.csv")
    parser.add_argument("--infer_path", type=str, default="output/topics_infer.csv")
    parser.add_argument("--test_output", type=str, default="output/topics_test.csv")
    parser.add_argument("--predict_output", type=str, default="output/topic_predict.csv")
    parser.add_argument("--num_topics", type=int, default=10, help="number of topics")
    parser.add_argument("--num_words", type=int, default=6, help="number of words per topic")
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--do_predict", default=False, action="store_true")
    parser.add_argument("--dictionary_file", type=str, default="pick/dictionary.pkl")
    parser.add_argument("--lda_file", type=str, default="pick/lda.pkl")
    args = parser.parse_args()

    if args.do_train:
        docs = load_documents(args.input_path)
        train_docs = pre_process(docs, args.stopwords_path)
        dictionary = construct_dictionary(train_docs)
        lda = train(dictionary, train_docs, args.topics_path, args.infer_path, args.num_topics, args.num_words)
        with open(args.dictionary_file, "wb") as f:
            pickle.dump(dictionary, f)
        with open(args.lda_file, "wb") as f:
            pickle.dump(lda, f)
    else:
        with open(args.dictionary_file, "rb") as f:
            dictionary = pickle.load(f)
        with open(args.lda_file, "rb") as f:
            lda = pickle.load(f)

    if args.do_test:
        test_docs = load_documents(args.test_path)
        test_docs = pre_process(test_docs, args.stopwords_path)
        test(lda, dictionary, test_docs, args.test_output)
    
    if args.do_predict:
        pred_doc = load_documents(args.predict_path)
        pred_doc = pre_process(pred_doc, args.stopwords_path)[0]
        predict(lda, dictionary, pred_doc, args.predict_output)


if __name__ == "__main__":
    main()
