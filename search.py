import random
import re
import numpy as np
import pandas as pd
import multiprocessing as mp
import nltk
from nltk import wordnet, pos_tag, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

class Document:
    def __init__(self, title, text):
        self.title = title
        self.text = text

    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text + ' ...']


index = []
reverse_index = {}
tf_idf_vectorizer = TfidfVectorizer()

def get_wordnet_pos(treebank_tag):
    my_switch = {
        'J': 'a',
        'V': 'v',
        'N': 'n',
        'R': 'r',
    }
    for key, item in my_switch.items():
        if treebank_tag.startswith(key):
            return item
    return 'n'


def my_lemmatizer(sent):
    lemmatizer = WordNetLemmatizer()
    tokenized_sent = sent.split()
    pos_tagged = [(word, get_wordnet_pos(tag))
                 for word, tag in pos_tag(tokenized_sent)]
    return ' '.join([lemmatizer.lemmatize(word, tag)
                    for word, tag in pos_tagged])


def text_lemmatizer(s):
    s = re.sub('[^a-z0-9\s]', '', s.lower())
    return my_lemmatizer(s)


def build_index(index=index, reverse_index=reverse_index, tf_idf_vectorizer=tf_idf_vectorizer):
    
    def build_document(id):
        return Document(df['name'][id], df['item_description'][id])
    
    # считывает данные и строит индекс
    df = pd.read_csv('train.tsv', sep='\t')
    delete = ['train_id', 'item_condition_id', 'category_name', 'brand_name', 'price', 'shipping']
    df = df.drop(columns=delete)
    df = df.dropna()
    df = df.reset_index(drop=True)
    index += list(map(build_document, [i for i in range(df.shape[0])]))
    file = open('dict.txt')
    dict_for_words = file.read().split('\n')
    file.close()
    for line in dict_for_words:
        if len(line) != 0:
            word, id_list = line.split(':')
            reverse_index[word] = []
            for num in id_list.split():
                if num != " ":
                    reverse_index[word].append(int(num))         
    file = open('lemmatizer.txt')
    texts = file.read().split('\n')
    file.close()
    tf_idf = tf_idf_vectorizer.fit_transform(texts)                    


def score(query, document):
    # возвращает скор для пары запрос-документ
    # больше -- релевантнее
    query_tf_idf = tf_idf_vectorizer.transform([query]).todense()
    document_tf_idf = tf_idf_vectorizer.transform([document.title + " " + document.text]).todense()
    len_query = (np.array(query_tf_idf)[0] ** 2).sum()
    len_document = (np.array(document_tf_idf)[0] ** 2).sum()
    if len_query == 0 or len_document == 0:
        return 0
    dist_cos = (np.array(query_tf_idf)[0]).dot(np.array(document_tf_idf)[0]) / np.sqrt(len_query) / np.sqrt(len_document)
    return 1 - abs(dist_cos)


def retrieve(query, reverse_index=reverse_index):
    # возвращает начальный список релевантных документов
    # реализация инвертированного индекса
    words = list(set(text_lemmatizer(query).split()))
    indicators = {}
    for word in words:
        indicators[word] = 0
    id_now = -1
    candidates = []
    not_empty = True
    while not_empty and len(indicators) != 0:
        check = True
        for word in indicators:
            if word not in reverse_index:
                check = False
                not_empty = False
                break                
            id_word = reverse_index[word][indicators[word]]
            if id_now == -1:
                id_now = id_word
            cnt_id = len(reverse_index[word])
            while id_now > id_word and indicators[word] < cnt_id - 1:
                indicators[word] += 1
                id_word = reverse_index[word][indicators[word]]
            if id_now < id_word:
                check = False
                id_now = id_word
            if indicators[word] == cnt_id - 1 and id_now != id_word:
                not_empty = False
                break
        if check:
            candidates.append(index[id_now])
            id_now += 1
    return candidates[:50]