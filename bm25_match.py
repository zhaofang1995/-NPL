'''
Created on 2019年4月28日

@author: lij
'''

import math
import jieba

class BM25(object):

    def __init__(self, docs):
        self.D = len(docs)
        self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / self.D
        self.docs = docs
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {}  # 存储每个词及出现了该词的文档数量
        self.idf = {}  # 存储每个词的idf值
        self.k1 = 1.5
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D - v + 0.5) - math.log(v + 0.5)

    def sim(self, doc, index):
        # 计算当前doc与语料库中编号为index的文档的相似度
        doc = list(jieba.cut(doc))
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word] * self.f[index][word] * (self.k1 + 1)
                      / (self.f[index][word] + self.k1 * (1 - self.b + self.b * d
                                                          / self.avgdl)))
        return score

    def sim_topK(self, doc, top_k):
        # 计算相似度最高的前K个
        tmp = {}
        for i in range(self.D):
            tmp[self.sim(doc, i)] = i
        max_sim = max(list(tmp.keys()))
        min_sim = min(list(tmp.keys()))
        scores = {}
        for w in tmp.values():
            scores[(self.sim(doc, w) - min_sim) / (max_sim - min_sim)] = w
        return sorted(scores.items(), key=lambda item: item[0], reverse=True)[:top_k]

    def sim_two(self, doc, index):
        tmp = {}
        for i in range(self.D):
            tmp[self.sim(doc, i)] = i
        max_sim = max(list(tmp.keys()))
        min_sim = min(list(tmp.keys()))
        return (self.sim(doc, index) - min_sim) / (max_sim - min_sim)




if __name__ == '__main__':
    """数据准备"""
    f = open('../data/allqa926_q.txt', encoding="utf-8")
    line = f.readline()
    text = []
    while line:
        # print(line)
        line = f.readline()
        text.append(line)
    f.close()
    
    # sents = utils.get_sentence(text)
    docs = []
    for sent in text:
        words = list(jieba.cut(sent))
        # words = utils.filter_stop(words)
        docs.append(words)
    # print(doc)

    bm25 = BM25(docs)
    # print(bm25.f)
    # print(bm25.idf)
    print(bm25.sim_two(doc="发票的种类有哪些？", index=1))
    print(bm25.sim_topK(doc="发票都有哪几种？", top_k=5))