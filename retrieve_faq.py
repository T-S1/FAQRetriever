import pickle
import numpy as np
import pandas as pd
import time
from scipy.stats import pearsonr
import io, sys
import pdb

from embed import calc_embedding
from mecab_parse import get_nouns



display_num = 5
search_num = 10
words_num = 6

dataset_path = 'dataset.csv'
embedded_docs_path = 'embedded_doc.pickle'
idf_path = 'idf.pickle'
nouns_path = 'nouns.pickle'

id_column = 'ID'
question_column = 'サンプル 問い合わせ文'
answer_column = 'サンプル 応答文'

df = pd.read_csv(dataset_path, index_col=0)
with open(embedded_docs_path, 'rb') as f:
    dic = pickle.load(f)
with open(idf_path, 'rb') as f:
    dic_idf = pickle.load(f)
with open(nouns_path, 'rb') as f:
    dic_nouns = pickle.load(f)


def print_results(results):
    print('\n----+----+----+----+---- Result ----+----+----+----+----')
    for i, result in enumerate(results[:display_num]):
        id = result[1]
        title = df[question_column][id]

        print(f'\n{i+1:>2}. {result[0]:.4f} {title}')
    print('\n----+----+----+----+----------------+----+----+----+----')


def print_words(max_words):
    print('\n----+----+----+----+----------------+----+----+----+----')
    print(f'\n関連する単語はどれですか？')
    for i, word in enumerate(max_words):
        print(f'\n{i+1:>2}. {word}')
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

    return results[:search_num]


def main():
    cnt_results = 1
    cnt_pair = 1
    nouns_set = {}

    print('\n検索文を入力してください。')
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
    query = sys.stdin.readline()

    start = time.time()

    done_set = set(get_nouns(query))

    while True:
        append_word = ''

        emb_query = calc_embedding(query)

        results = search_results(emb_query)

        print_results(results)
        # print(cnt_results, '回目')
        cnt_results += 1

        end = time.time()
        print(f'検索時間: {end - start} s')

        if select_results(results):
            return

        # 表示した項目を削除
        for result in results[:display_num]:
            id = result[1]
            dic.pop(id)

        while True:
            ids_high = []
            idfs_high = []
            idfs_nouns = []
            max_idfs = []
            max_words = []
            

            if len(idfs_nouns) < words_num:

                # 上位の結果内に関連項目がない場合、削除
                for id in ids_high:
                    dic.pop[id]

                results = search_results(emb_query)
                ids_high = [id for _, id in results]

                nouns_list = []
                for id in ids_high:
                    nouns_list.extend(dic_nouns[id])

                nouns_set = set(nouns_list) - done_set
                nouns_set = sorted(nouns_set, key=nouns_list.index, reverse=True)

                for noun in nouns_set:
                    idfs_high.append(dic_idf[noun])

                idfs_nouns = sorted(zip(idfs_high, nouns_set), key=lambda x: x[0])
                # pdb.set_trace()

            for i in range(words_num):
                idf, noun = idfs_nouns.pop()
                max_idfs.append(idf)
                max_words.append(noun)

            print_words(max_words)
            # print(cnt_pair, '回目')

            print('\n該当する単語がある場合はその番号、ない場合は 0 を入力')

            option = sys.stdin.readline()

            for i in range(words_num):
                if option == f'{i+1}\n':
                    append_word = max_words[i]
                done_set.add(max_words[i])

            if append_word != '':
                break

            cnt_pair += 1

        start = time.time()

        query += f' {append_word}'

        # pdb.set_trace()


if __name__ == '__main__':
    main()
