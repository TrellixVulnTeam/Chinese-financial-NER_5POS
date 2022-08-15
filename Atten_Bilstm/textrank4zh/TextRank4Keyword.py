# -*- encoding:utf-8 -*-
"""
@author:   letian
@homepage: http://www.letiantian.me
@github:   https://github.com/someus/
"""

from . import util
from Atten_Bilstm.textrank4zh.Segmentation import Segmentation


class TextRank4Keyword(object):

    def __init__(self, stop_words_file=None,
                 allow_speech_tags=util.allow_speech_tags,
                 delimiters=util.sentence_delimiters):
        """
        Keyword arguments:
        stop_words_file  --  str，指定停止词文件路径（一行一个停止词），若为其他类型，则使用默认停止词文件
        delimiters       --  默认值是`?!;？！。；…\n`，用来将文本拆分为句子。
        
        Object Var:
        self.words_no_filter      --  对sentences中每个句子分词而得到的两级列表。
        self.words_no_stop_words  --  去掉words_no_filter中的停止词而得到的两级列表。
        self.words_all_filters    --  保留words_no_stop_words中指定词性的单词而得到的两级列表。
        """
        self.text = ''
        self.keywords = None

        self.seg = Segmentation(stop_words_file=stop_words_file,
                                allow_speech_tags=allow_speech_tags,
                                delimiters=delimiters)

        self.sentences = None
        self.words_no_filter = None  # 2维列表
        self.words_no_stop_words = None
        self.words_all_filters = None

    def analyze(self, text,
                window=2,
                lower=False,
                vertex_source='all_filters',
                edge_source='no_stop_words',
                pagerank_config={'alpha': 0.85, }):
        """分析文本

        Keyword arguments:
        text       --  文本内容，字符串。
        window     --  窗口大小，int，用来构造单词之间的边。默认值为2。
        lower      --  是否将文本转换为小写。默认为False。
        vertex_source   --  选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点。
                            默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。关键词也来自`vertex_source`。
        edge_source     --  选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点之间的边。
                            默认值为`'no_stop_words'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。边的构造要结合`window`参数。
        """

        # self.text = util.as_text(text)
        self.text = text
        self.word_index = {}
        self.index_word = {}
        self.keywords = []
        self.graph = None

        result = self.seg.segment(text=text, lower=lower)
        self.sentences = result.sentences
        self.words_no_filter = result.words_no_filter
        self.words_no_stop_words = result.words_no_stop_words
        self.words_all_filters = result.words_all_filters

        util.debug(20 * '*')
        util.debug('self.sentences in TextRank4Keyword:\n', ' || '.join(self.sentences))
        util.debug('self.words_no_filter in TextRank4Keyword:\n', self.words_no_filter)
        util.debug('self.words_no_stop_words in TextRank4Keyword:\n', self.words_no_stop_words)
        util.debug('self.words_all_filters in TextRank4Keyword:\n', self.words_all_filters)

        options = ['no_filter', 'no_stop_words', 'all_filters']

        if vertex_source in options:
            _vertex_source = result['words_' + vertex_source]
        else:
            _vertex_source = result['words_all_filters']

        if edge_source in options:
            _edge_source = result['words_' + edge_source]
        else:
            _edge_source = result['words_no_stop_words']

        self.keywords = util.sort_words(_vertex_source, _edge_source, window=window, pagerank_config=pagerank_config)

    def get_keywords(self, num=6, word_min_len=1):
        """获取最重要的num个长度大于等于word_min_len的关键词。

        Return:
        关键词列表。
        """
        result = []
        count = 0
        for item in self.keywords:
            if count >= num:
                break
            if len(item.word) >= word_min_len:
                result.append(item)
                count += 1
        return result

    def sentences_similarity(self, input_sentence, num, sentence=False):
        """
        计算一个句子与文章中其他句子的相似度

        input_sentence       --  代表一个句子，是由句子组成的列表
        num                  --  取和输入句子相似度前num的句子输出
        sentence             --  返回句子，否则即返回单词列表
        """

        # 进行一些初始化
        sentences_dict = {}
        result = {}
        is_in_acticle = False
        sentences_index = -1

        # 首先判断该句子是否在文章中
        for i in range(len(self.sentences)):
            if input_sentence in self.sentences[i]:
                is_in_acticle = True
                sentences_index = i
        if is_in_acticle:
            print("读取数据中。。。")
        else:
            print("input error")
            return

        # 判断句子在文章中，开始进行遍历求值
        sentences_word = self.words_all_filters[sentences_index]
        for i in range(len(self.sentences)):
            if i == sentences_index:
                continue

            sentences_dict[i] = util.get_similarity(
                self.words_all_filters[i],
                sentences_word)

        # 接下来对相似度进行排序，获取相似度最高的几个句子

        result = sorted(sentences_dict.items(), key=lambda item: item[1], reverse=True)

        result = result[:num]

        dictdata = {}
        for i in result:
            dictdata[i[0]] = i[1]

        # return_dict = {}
        # if sentence:
        #     for key, value in dictdata:
        #         return_dict[self.sentences[key]] = value
        #     return return_dict
        # else:
        #     for key, value in dictdata:
        #         return_dict[self.words_all_filters[key]] = value
        return dictdata

    def get_keyphrases(self, keywords_num=12, min_occur_num=2):
        """获取关键短语。
        获取 keywords_num 个关键词构造的可能出现的短语，要求这个短语在原文本中至少出现的次数为min_occur_num。

        Return:
        关键短语的列表。
        """
        keywords_set = set([item.word for item in self.get_keywords(num=keywords_num, word_min_len=1)])
        keyphrases = set()
        for sentence in self.words_no_filter:
            one = []
            for word in sentence:
                if word in keywords_set:
                    one.append(word)
                else:
                    if len(one) > 1:
                        keyphrases.add(''.join(one))
                    if len(one) == 0:
                        continue
                    else:
                        one = []
            # 兜底
            if len(one) > 1:
                keyphrases.add(''.join(one))

        return [phrase for phrase in keyphrases
                if self.text.count(phrase) >= min_occur_num]


if __name__ == '__main__':
    pass
