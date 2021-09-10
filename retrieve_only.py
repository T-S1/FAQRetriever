import pickle
import numpy as np
import pandas as pd
import time
from scipy.stats import pearsonr
import io, sys
import pdb

from embed import calc_embedding


display_num = 20

dataset_path = 'dataset.csv'
embedded_docs_path = 'embedded_doc.pickle'

id_column = 'ID'
question_column = 'サンプル 問い合わせ文'
answer_column = 'サンプル 応答文'

df = pd.read_csv(dataset_path, index_col=0)
with open(embedded_docs_path, 'rb') as f:
    dic = pickle.load(f)


def print_results(results):
    print('\n----+----+----+----+---- Result ----+----+----+----+----')
    for i, result in enumerate(results[:display_num]):
        id = result[1]
        # title = df[question_column][df[id_column] == id].values[0]
        title = df[question_column][id]

        print(f'\n{i+1:>2}. {result[0]:.4f} {title}')
    print('\n----+----+----+----+----------------+----+----+----+----')


def select_results(results):
    while True:
        print('\n該当する結果がある場合はその番号、ない場合は 0 を入力')
        option = sys.stdin.readline()
        # print(list(option))
        if option == '0\n':
            return False
        for i in range(display_num):
            if option == f'{i+1}\n':
                id = results[i][1]
                question = df[question_column][id]
                answer = df[answer_column][id]
                print('\n----+----+----+----+----------------+----+----+----+----')
                print(f'\nQ. {question}')
                print(f'\nA. {answer}')
                print('\n----+----+----+----+----------------+----+----+----+----')
                return True


def calc_dst_colbert(emb_query, emb_doc):
    """
    Args:
        emb_query (numpy.array): shape (words of query, dims)
        emb_doc (numpy.array): shape (words of doc, dims)
    """
    score = 0
    for vec_query in emb_query:
        rs = []

        for vec_doc in emb_doc:
            # pdb.set_trace()
            r = pearsonr(vec_query, vec_doc)[0]
            rs.append(r)

        score += max(rs)

    return 1 - score / emb_query.shape[0]


def search_results(emb_query):
    ids = []
    dsts = []

    for id in dic:
        emb_answer = dic[id]

        dst = calc_dst_colbert(emb_query, emb_answer)

        ids.append(id)
        dsts.append(dst)

    results = sorted(zip(dsts, ids))

    return results[:display_num]


def main():
    print('\n検索文を入力してください。')
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
    query = sys.stdin.readline()

    start = time.time()

    emb_query = calc_embedding(query)

    results = search_results(emb_query)

    print_results(results)

    end = time.time()
    print(f'検索時間: {end - start} s')

    select_results(results)

    # pdb.set_trace()


if __name__ == '__main__':
    main()
