# -*- encoding:utf-8 -*-


import codecs
from Atten_Bilstm.textrank4zh import TextRank4Keyword, TextRank4Sentence


def main(file_name, sen, num=5, return_sentence=False):
    text = codecs.open(file_name, 'r', 'utf-8').read()
    tr4w = TextRank4Keyword()

    tr4w.analyze(text=text, lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象

    # sen1 = '30日中午，有媒体曝光高圆圆和赵又廷现身台北桃园机场的照片，照片中两人小动作不断，尽显恩爱'

    # 返回与sen1相关度最高的num个句子的word_list
    try:
        sentence_rank = tr4w.sentences_similarity(sen, num, return_sentence)
    except:
        pass

    sentences_vector = []
    sentences_weight = []
    result = {}

    # 开始计算全文权重
    # print(sentence_rank)
    # for index, weight in sentence_rank.items():
    #     for each in tr4w.words_all_filters[index]:
    #         sentences_vector.append(each)
    #         sentences_weight.append(weight)

    return sentences_vector, sentences_weight


if __name__ == '__main__':
    # 获取到相关词语机器权值
    sen = '30日中午，有媒体曝光高圆圆和赵又廷现身台北桃园机场的照片，照片中两人小动作不断，尽显恩爱'
    print(main("data.txt",sen))
