import torch
from transformers import BertJapaneseTokenizer, BertModel
import numpy as np
import pdb


# モデルの読み込み
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model_bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model_bert.eval()


def average_emv(emb):
    return np.average(emb, axis=1)


def calc_embedding(text):
    """ テキストのエンコード
    Args:
        text (str): 入力文字列
    Returns:
        エンコードした1単語目（[CLS]）のベクトル
    """
    bert_tokens = tokenizer.tokenize(text)

    ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + bert_tokens[:126] + ["[SEP]"])
    tokens_tensor = torch.tensor(ids).reshape(1, -1)

    with torch.no_grad():
        output = model_bert(tokens_tensor)

    emb = output[0].numpy()

    # emb = average_emv(emb)


    return emb[0]


# text = '母子手帳の受け取り場所はどこですか？'
# result = calc_embedding(text)
# print(result)
# print(result.shape)
