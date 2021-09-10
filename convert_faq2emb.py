import pickle
import pandas as pd
import numpy as np
import time
import pdb

from embed import calc_embedding


dataset_path = 'dataset.csv'
output_path = 'embedded_doc.pickle'

id_column = 'ID'
doc_column = 'サンプル 問い合わせ文'


def main():
    df = pd.read_csv(dataset_path)
    dic = {}

    for i in range(len(df)):
        id = df[id_column][i]
        doc = df[doc_column][i]
        assert id not in dic, '\n重複するIDが存在します'

        emb_doc = calc_embedding(doc)
        dic[id] = emb_doc

        print(f'Done {i+1:03} / {len(df)}\r', end='')

        # if i == 2:
        #     print('\nShape:', embedded_answer.shape)
        #     break

    with open(output_path, 'wb') as f:
        pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

    # pdb.set_trace()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'\nTime: {end - start} s')
