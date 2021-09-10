import MeCab


mecab_tagger = MeCab.Tagger("-Ochasen")


def tokenize(text):
    parsed = mecab_tagger.parse(text).splitlines()[:-1]
    tokens = [chunk.split('\t')[0] for chunk in parsed]
    return tokens


def get_nouns(text):
    parsed = mecab_tagger.parse(text).splitlines()[:-1]
    tokens = [chunk.split('\t')[0] for chunk in parsed]
    word_classes = [chunk.split('\t')[3].split('-')[0] for chunk in parsed]
    nouns = [tokens[i] for i, word_class in enumerate(word_classes) if word_class == '名詞']
    return nouns
