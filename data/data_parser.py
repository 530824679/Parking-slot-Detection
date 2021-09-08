
import os
from tqdm import tqdm
import argparse
import xml.etree.ElementTree as ET

sets = ['train', 'val', 'test']
classes = ['car', 'slot', 'corner']

class ParserXML(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_dir = os.path.join(data_dir, 'labels')
        self.image_dir = os.path.join(data_dir, 'images')
        self.xml_dir = os.path.join(data_dir, 'xml')

    def parser(self):

        for image_set in sets:
            if not os.path.exists(self.label_dir):
                os.makedirs(self.label_dir)
            image_ids = open('{}/{}.txt'.format(self.label_dir, image_set)).read().strip().split()
            list_file = open('{}/{}.txt'.format(self.data_dir, image_set), 'w')
            for image_id in tqdm(image_ids):
                list_file.write('{}/{}.png\n'.format(self.image_dir, image_id))
                self.convert_xml2txt(image_id)
            list_file.close()

    def convert(self, size, box):
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0 - 1
        y = (box[2] + box[3]) / 2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    def convert_xml2txt(self, image_id):
        try:
            in_file = open('{}/{}.xml'.format(self.xml_dir, image_id), encoding='utf-8')
            out_file = open('{}/{}.txt'.format(self.label_dir, image_id), 'w', encoding='utf-8')
            tree = ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')

            w = int(size.find('width').text)
            h = int(size.find('height').text)
            c = int(size.find('depth').text)

            for obj in root.iter('object'):
                cls = obj.find('class').text
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                b1, b2, b3, b4 = b
                # 标注越界修正
                if b2 > w:
                    b2 = w
                if b4 > h:
                    b4 = h
                b = (b1, b2, b3, b4)
                bb = self.convert((w, h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        except Exception as e:
            print(e, image_id)


if __name__ == '__main__':
    data_dir = '/home/chenwei/HDD/Project/3DVD/datasets'
    xml = ParserXML(data_dir)
    xml.parser()


