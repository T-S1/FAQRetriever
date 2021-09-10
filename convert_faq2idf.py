import pickle
import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
import pdb

from mecab_parse import get_nouns


dataset_path = 'dataset.csv'
output_path = 'idf.pickle'
nouns_path = 'nouns.pickle'

id_column = 'ID'
doc_column = 'サンプル 問い合わせ文'


def main():
    df = pd.read_csv(dataset_path)
    ids = []
    nouns_list = []
    dic = {}
    dic_nouns = {}

    for i in range(len(df)):
        id = df[id_column][i]
        doc = df[doc_column][i]
        assert id not in ids, '\n重複するIDが存在します'

        nouns = get_nouns(doc)

        ids.append(id)
        nouns_list.append(' '.join(nouns))
        dic_nouns[id] = nouns

        print(f'Done {i+1:03} / {len(df)}\r', end='')

        # if i == 2:
            
        #     print(nouns_list)
        #     break

    # bag of words
    docs = np.array(nouns_list)
    vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
    bow = vectorizer.fit_transform(docs)
    flgs = (bow.toarray() >= 1)
    cnts = np.sum(flgs, axis=0)

    docs_num = len(bow.toarray())

    words = vectorizer.get_feature_names()

    # idf
    idf = np.log(docs_num / (1 + cnts))

    print(f'\n{len(words)} words')
    print('idf shape:', idf.shape)

    for i in range(len(idf)):
        word = words[i]
        x = idf[i]
        dic[word] = x

    with open(output_path, 'wb') as f:
        pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

    with open(nouns_path, 'wb') as f:
        pickle.dump(dic_nouns, f, pickle.HIGHEST_PROTOCOL)

    pdb.set_trace()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'\nTime: {end - start} s')
