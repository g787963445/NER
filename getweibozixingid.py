# from tqdm import tqdm
#
# from hanzi_chaizi_ import HanziChaizi
#
# def is_Chinese(ch):
#
#     if '\u4e00' <= ch <= '\u9fff':
#         return True
#     return False
#
# hc = HanziChaizi()
#
# f1 = "datas/data_sum.txt"
#
#
# def read_file(filename):
#     X, y, Z = [], [], []
#     # x是字，y是标签，z是部首
#     labels = []
#     with open(filename, 'r', encoding='utf-8') as f:
#         x0, y0, z0 = [], [], []
#         for line in f:
#             data = line.strip()
#             if data:
#                 x0.append(data.split()[0])
#                 z0.append(data.split()[1])
#                 y0.append(data.split()[-1])
#
#             else:
#                 if len(x0) != 0:
#                     X.append(x0)
#                     y.append(y0)
#                     Z.append(z0)
#                 x0, y0, z0 = [], [], []
#         if len(x0) != 0:
#             X.append(x0)
#             y.append(y0)
#             Z.append(z0)
#     return X, y, Z
#
# x,y,z = read_file(f1)
#
#
#
#
#
#
# #一个部首对应一个id
# def zixingtoid():
#     num = 1
#     zixing2id = {}
#     id2zixing = {}
#     zixing_ids = []
#     x,y,z =read_file(f1)
#     for lines in z:
#         for zixings in lines:
#             for zixing in zixings:
#                 zixing = zixing.strip()
#                 if zixing not in zixing2id:
#                     zixing2id[zixing] = num
#                     id2zixing[num] = zixing
#                     num += 1
#                     zixing_ids.append(num)
#
#
#     return zixing2id,id2zixing
# # #
#
# zixing2id,id2zixing = zixingtoid()
# # print(zixing2id)
#
# #纯id
# # def getzixing_id():
# #     num = 1
# #     zixing2id = {}
# #     id2zixing = {}
# #     zixing_ids = []
# #     x, y, z = read_file(f1)
# #     for lines in z:
# #         for zixings in lines:
# #             for zixing in zixings:
# #                 zixing = zixing.strip()
# #                 if zixing not in zixing2id:
# #                     zixing2id[zixing] = num
# #                     id2zixing[num] = zixing
# #                     zixing_ids.append(num)
# #                     num += 1
# #     return zixing_ids
#
# # 一个字对应多个部首
# def wordtozixing():
#     word2zixing = {}
#     zixing2word = {}
#     for i, sequence in tqdm(enumerate(x)):
#         for line in sequence:
#             lines = line.strip()
#             zixing = ""
#             if is_Chinese(lines):
#                 try:
#                     zixings = hc.query(lines)
#                     for i in zixings:
#                         zixing += i
#                     word2zixing[lines] = zixing
#                     zixing2word[zixing] = lines
#                 except TypeError:
#                     zixing = lines
#                     word2zixing[lines] = zixing
#                     zixing2word[zixing] = lines
#             else:
#                 zixing = lines
#                 word2zixing[lines] = zixing
#                 zixing2word[zixing] = lines
#     return word2zixing, zixing2word
#
# word2zixing,zixing2word = wordtozixing()
# # print(word2zixing)
#
# def getzixing_id():
#     zixing_ids = []
#     word2id = {}
#
#     word2zixing, _ = wordtozixing()
#     # print(word2zixing)
#
#     for k, v in word2zixing.items():
#         zixing_id = []
#         for j in v.strip("\n"):
#             zixing_id.append(zixing2id[j])
#         while len(zixing_id)<16:
#             zixing_id.append(0)
#         if k not in word2id:
#             word2id[k] = zixing_id
#         if k not in zixing_id:
#             zixing_ids.append(zixing_id)
#
#     return zixing_ids
#
# # zixing_ids = getzixing_id()
# # print(zixing_ids)
#
# def get_word_to_id():
#     zixing_ids = []
#     word2id = {}
#
#     word2zixing, _ = wordtozixing()
#     # print(word2zixing)
#
#     for k, v in word2zixing.items():
#         zixing_id = []
#         for j in v.strip("\n"):
#             zixing_id.append(zixing2id[j])
#         while len(zixing_id)<16:
#             zixing_id.append(0)
#         if k not in word2id:
#             word2id[k] = zixing_id
#
#         if k not in zixing_id:
#             zixing_ids.append(zixing_id)
#
#     return word2id
# #
# # word2id = get_word_to_id()
# # print(word2id)
# # def juzizixing_id():
# zixinggetid = []
# for i, sequence in tqdm(enumerate(x)):
#     for line in sequence:
#         lines = line.strip()
#         word_id = []


from tqdm import tqdm

from hanzi_chaizi_ import HanziChaizi

def is_Chinese(ch):

    if '\u4e00' <= ch <= '\u9fff':
        return True
    return False

hc = HanziChaizi()

f1 = "datas/data_sum.txt"


def read_file(filename):
    X, y, Z = [], [], []
    # x是字，y是标签，z是部首
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        x0, y0, z0 = [], [], []
        for line in f:
            data = line.strip()
            if data:
                x0.append(data.split()[0])
                z0.append(data.split()[1])
                y0.append(data.split()[-1])

            else:
                if len(x0) != 0:
                    X.append(x0)
                    y.append(y0)
                    Z.append(z0)
                x0, y0, z0 = [], [], []
        if len(x0) != 0:
            X.append(x0)
            y.append(y0)
            Z.append(z0)
    return X, y, Z

x,y,z = read_file(f1)




#一个部首对应一个id
def zixingtoid():
    num = 1
    zixing2id = {}
    id2zixing = {}
    zixing_ids = []
    x,y,z =read_file(f1)
    for lines in z:
        for zixings in lines:
            for zixing in zixings:
                zixing = zixing.strip()
                if zixing not in zixing2id:
                    zixing2id[zixing] = num
                    id2zixing[num] = zixing
                    num += 1
                    zixing_ids.append(num)


    return zixing2id,id2zixing
# #

# zixing2id,id2zixing = zixingtoid()
# print(zixing2id)

def wordtozixing():
    word2zixing = {}
    zixing2word = {}
    for i, sequence in tqdm(enumerate(x)):
        for line in sequence:
            lines = line.strip()
            zixing = ""
            if is_Chinese(lines):
                try:
                    zixings = hc.query(lines)
                    for i in zixings:
                        zixing += i
                    word2zixing[lines] = zixing
                    zixing2word[zixing] = lines
                except TypeError:
                    zixing = lines
                    word2zixing[lines] = zixing
                    zixing2word[zixing] = lines
            else:
                zixing = lines
                word2zixing[lines] = zixing
                zixing2word[zixing] = lines
    return word2zixing, zixing2word

word2zixing,zixing2word = wordtozixing()
zixing2id,id2zixing = zixingtoid()
def wordtranszixing(sequence):

    zixings = []
    for line in sequence:
        lines = line.strip()
        zixing = word2zixing[lines]
        zixings.append(zixing)
    return zixings

def wordtransid(sequence):
    zixings = wordtranszixing(sequence)
    # print(word2zixing)
    zixing_ids = []
    word2id = {}
    for line in zixings:
        zixing_id = []
        line = line.strip()
        for k in line:
            zixing_id.append(zixing2id[k])
        while(len(zixing_id)<16):
            zixing_id.append(0)
        zixing_ids.append(zixing_id)
    return zixing_ids

def get_all_word2id():
    all_zixing_ids = []
    for i, sequence in tqdm(enumerate(x)):
        zixings = wordtranszixing(sequence)
        # print(word2zixing)
        zixing_ids = []
        word2id = {}
        for line in zixings:
            zixing_id = []
            line = line.strip()
            for k in line:
                zixing_id.append(zixing2id[k])
            while (len(zixing_id) < 16):
                zixing_id.append(0)
            zixing_ids.append(zixing_id)
        all_zixing_ids.append(zixing_ids)
    return all_zixing_ids

def produce_zixing_id_mask(zixing_ids):
    mask_sum = []
    for line in zixing_ids:
        mask = []
        for word in line:
            small_mask = []
            for i in word:
                if i!=0:
                    small_mask.append(1)
                else:
                    small_mask.append(0)
            mask.append(small_mask)
        mask_sum.append(mask)
    return mask_sum