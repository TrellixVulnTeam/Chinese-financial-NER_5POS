# -*- encoding:utf-8 -*-
# 本代码实现：
# 输入句子及其所在文章，获取相似度最高的num个句子，并对其进行分类处理
import Atten_Bilstm.classifier_main as main1
import Atten_Bilstm.text_rank_main as main2


def blstm(article_dir, input_sentence):
    """
    首先输入文章，对文章进行处理

    article_dir     --  文章路径
    input_sentence  --  输入句子（注意，该句子必须存在于文章中，否则会报错）
    """

    # 获取相关词语分词结果及权值
    word_list, weight_list = main2.main(article_dir, input_sentence)

    # 将该关系存入txt文件，方便之后调用
    with open("analyze_result.txt", "w", encoding="utf8") as f:
        for i in range(len(word_list)):
            f.write(f"{word_list[i]}    {weight_list[i]}\n")
        f.write("\n")

    return main1.main()

    # 输入模型进行分类


if __name__ == '__main__':
    sen = '30日中午，有媒体曝光高圆圆和赵又廷现身台北桃园机场的照片，照片中两人小动作不断，尽显恩爱'
    print(run("data.txt", sen))
