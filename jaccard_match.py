'''
Created on 2019年4月28日

@author: lij
'''

import jieba



class TF_IDF_JAC():
    def __init__(self):
        self.docs = {}
        self.seg_docs = self.get_seg_docs()
        self.stopword = []

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

    def jaccard_sim(self, doc1, doc2):

        unions = len(set(doc1).union(set(doc2)))
        intersections = len(set(doc1).intersection(set(doc1)))
        return 1. * intersections / unions

    def jaccard_two(self, doc1, doc2):

        doc1 = [w for w in jieba.lcut(doc1) if len(w) > 1 and w not in self.stopword and w.isalpha()]
        doc2 = [w for w in jieba.lcut(doc2) if len(w) > 1 and w not in self.stopword and w.isalpha()]

        return self.jaccard_sim(doc1, doc2)

    def jaccard_topK(self, doc, top_k):
        doc = [w for w in jieba.lcut(doc) if len(w) > 1 and w not in self.stopword and w.isalpha()]

        scores = {}
        dict_docs = dict(zip(list(range(len(self.seg_docs))), self.seg_docs))
        for key, value in dict_docs.items():
            scores[self.jaccard_sim(doc, value)] = key
        return sorted(scores.items(), key=lambda item: item[0], reverse=True)[:top_k]


if __name__ == '__main__':
    jac = TF_IDF_JAC()
    doc_ = "发票有哪几种？"
    doc_1 = "发票有哪几种？"
    doc_2 = "发票的种类有哪些？"
    # tf_idf.get_text_vector(doc1)
    
    print(jac.jaccard_two(doc_1, doc_2))
    print(jac.jaccard_topK(doc_, top_k=5))
