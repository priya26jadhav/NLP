import gensim
from gensim import corpora
text1 = [" Hello everyone, This is Natural Language Processing Laboratory, Natural Language Processing is branch of Artificial Intelligence"]
tokens = [[item for item in line.split()] for line in text1]
g_dict1 = corpora.Dictionary(tokens)
print("The dictionary has: " +str(len(g_dict1)) + " tokens\n")
print(g_dict1.token2id)


g_bow =[g_dict1.doc2bow(token, allow_update = True) for token in tokens]
print("Bag of Words : ", g_bow)

from gensim import models
import numpy as np
g_tfidf = models.TfidfModel(g_bow, smartirs='ntc')
print("TF-IDF Vector:")
for item in g_tfidf[g_bow]:
    print([[g_dict1[id], np.around(freq, decimals=2)] for id, freq in item])
    

from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count
import gensim.downloader as api
dataset = api.load("text8")
words = [d for d in dataset]

data1 = words[:1000]
w2v_model = Word2Vec(data1, min_count = 0, workers=cpu_count())
print(w2v_model.wv.most_similar('social'))

