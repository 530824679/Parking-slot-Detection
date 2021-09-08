import os
import sys
import random
import argparse

sys.path.append('..')

def split_data(xml_path, txt_path, trainval_percent, train_percent):
    total_xml = os.listdir(xml_path)
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    num = len(total_xml)
    list_index = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list_index, tv)
    train = random.sample(trainval, tr)

    file_trainval = open(txt_path + '/trainval.txt', 'w')
    file_test = open(txt_path + '/test.txt', 'w')
    file_train = open(txt_path + '/train.txt', 'w')
    file_val = open(txt_path + '/val.txt', 'w')

    for i in list_index:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            file_trainval.write(name)
            if i in train:
                file_train.write(name)
            else:
                file_val.write(name)
        else:
            file_test.write(name)

    file_trainval.close()
    file_train.close()
    file_val.close()
    file_test.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_path', default='/home/chenwei/HDD/Project/3DVD/datasets/xml', type=str, help='input xml label path')
    parser.add_argument('--txt_path', default='/home/chenwei/HDD/Project/3DVD/datasets', type=str, help='output txt label path')
    opt = parser.parse_args()

    trainval_percent = 1.0
    train_percent = 0.95

    xmlfilepath = opt.xml_path
    txtsavepath = opt.txt_path

    split_data(xmlfilepath, txtsavepath, trainval_percent, train_percent)