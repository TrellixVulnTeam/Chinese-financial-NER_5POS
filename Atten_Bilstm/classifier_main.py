import numpy as np
import torch
from Atten_Bilstm.data import build_corpus
from Atten_Bilstm.utils import extend_maps, prepocess_data_for_lstmcrf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 读取数据

    test_word_lists, test_tag_lists, word2id, tag2id = build_corpus("analyze_result")
    tagid = {'PRODUCT_NAME': 0, 'TIME': 1, 'O': 2, 'PERSON_NAME': 3, 'ORG_NAME': 4, 'LOCATION': 5,
             'COMPANY_NAME': 6}

    test_word_lists = [i for i in test_word_lists if i != []]
    test_tag_lists = [i for i in test_tag_lists if i != []]

    model = torch.load("./trained_model/bilstm_crf.pkl", map_location="cpu")

    # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
    crf_word2id, crf_tag2id = extend_maps(word2id, tagid, for_crf=True)
    print(' '.join([i[0] for i in crf_tag2id.items()]))

    # 还需要额外的一些数据处理
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists
    )

    pred_tag_lists = model.test(
        test_word_lists, test_tag_lists, word2id, tagid)

    pred_tag_lists = pred_tag_lists.tolist()
    sum_weight = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    # print(pred_tag_lists[0])

    for each in pred_tag_lists[0]:
        sum_weight[each] += 1

    result_list = list(sum_weight.values())
    # for i in range(len(result_list)):
    #     result_list[i] = result_list[i] / len(pred_tag_lists)

    return torch.tensor(result_list[:7])


if __name__ == '__main__':
    # 此文件只能在cuda环境中运行
    print(main())
