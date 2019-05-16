'''
Created on 2019年4月28日

@author: lij
'''

import jieba
import math


# import os
# import json

# from collections import OrderedDict

class TF_IDF():
    def __init__(self):
        self.docs = {}
        self.seg_docs = self.get_seg_docs()
        self.stopword = []
        self.tf = []
        self.df = {}
        self.idf = {}
        self.topK_idf = {}
        self.bow = {}
        self.cal_tfidf()

    def read_file(self, path):
        f = open(path, encoding="utf-8")
        line = f.readline()
        text = []
        while line:
            # print(line)
            line = f.readline()
            text.append(line)
        f.close()
        return text

    def get_seg_docs(self):
        _seg_docs = []

        DOCUMENT = './data/allqa926_q.txt'
        STOPWORD = './data/stop_words.txt'

        self.docs = self.read_file(DOCUMENT)
        with open(STOPWORD, 'r', encoding='utf-8') as file:
            self.stopword = file.read()
        for i in range(len(self.docs)):
            doc_seg = [w for w in jieba.lcut(self.docs[i]) if len(w) > 1 and w not in self.stopword and w.isalpha()]
            _seg_docs.append(doc_seg)
        return _seg_docs


    def cal_tfidf(self):
        # 统计词频
        for doc in self.seg_docs:
            bow = {}
            for word in doc:
                if not word in bow:
                    bow[word] = 0
                bow[word] += 1
            self.tf.append(bow)
            for word, _ in bow.items():
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1
        for word, df in self.df.items():
            # 只出现过在一篇文檔的词不要(选择性)
            if df < 2:
                pass
            else:
                self.idf[word] = math.log10(len(self.seg_docs) / df)

    def tf(self, index, word):
        return self.tf[index][word]

    def idf(self, word):
        return self.idf[word]

    def tf_idf(self, index, word):
        return self.tf[index][word] * self.idf[word]

    def get_text_vector(self, doc):

        # self.get_text_vector(doc_1)
        v_doc = []
        for w in doc:
            for word, value in self.idf.items():
                if w == word:
                    v_doc.append(value)
            else:
                v_doc.append(0)
        return v_doc

    def cosine_similarity(self, doc1, doc2):
        # compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
        # doc1= list(jieba.cut(doc1))

        sum_xx, sum_xy, sum_yy = 0.0, 0.0, 0.0
        for i in range(len(doc1)):
            x, y = doc1[i], doc2[i]
            sum_xx += math.pow(x, 2)
            sum_yy += math.pow(y, 2)
            sum_xy += x * y
        try:
            return sum_xy / math.sqrt(sum_xx * sum_yy)
        except ZeroDivisionError:
            return 0

    def cosine_two(self, doc1, doc2):

        doc1 = [w for w in jieba.lcut(doc1) if len(w) > 1 and w not in self.stopword and w.isalpha()]
        doc2 = [w for w in jieba.lcut(doc2) if len(w) > 1 and w not in self.stopword and w.isalpha()]

        v_doc1 = self.get_text_vector(doc1)
        v_doc2 = self.get_text_vector(doc2)
        for _ in range(int(math.fabs(len(v_doc1) - len(v_doc2)))):
            if len(v_doc1) < len(v_doc2):
                v_doc1.append(0)
            elif len(v_doc1) > len(v_doc2):
                v_doc2.append(0)
            else:
                break
        score1 = self.cosine_similarity(v_doc1, v_doc1)
        score2 = self.cosine_similarity(v_doc1, v_doc2)
        return 1 - math.fabs((score2 - score1) / score1)

    def cosine_topK(self, doc, top_k):
        scores = {}
        doc = [w for w in jieba.lcut(doc) if len(w) > 1 and w not in self.stopword and w.isalpha()]
        doc_v = self.get_text_vector(doc)
        dict_docs = dict(zip(list(range(len(self.seg_docs))), self.seg_docs))
        for key, value in dict_docs.items():
            value_v = self.get_text_vector(value)
            for _ in range(int(math.fabs(len(doc_v) - len(value_v)))):
                if len(doc_v) < len(value_v):
                    doc_v.append(0)
                elif len(doc_v) > len(value_v):
                    value_v.append(0)
                else:
                    break
            scores[self.cosine_similarity(doc_v, value_v)] = key
        return sorted(scores.items(), key=lambda item: item[0], reverse=True)[:top_k]


if __name__ == '__main__':
    tf_idf = TF_IDF()
    doc_ = "发票有哪几种？"
    doc_1 = "发票有哪几种？"
    doc_2 = "发票的种类有哪些？"
    # tf_idf.get_text_vector(doc1)
    
    print(tf_idf.cosine_two(doc_1, doc_2))
    print(tf_idf.cosine_topK(doc_, top_k=5))



