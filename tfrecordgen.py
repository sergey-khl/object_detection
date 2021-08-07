import os
import glob
import pandas as pd
import io
import argparse
import csv
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow csv-to-TFRecord converter")
parser.add_argument("-x",
                    "--csv_dir",
                    help="Path to the folder where the input .csv files are stored.",
                    type=str)
parser.add_argument("-l",
                    "--labels_path",
                    help="Path to the labels (.pbtxt) file.", type=str)
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output TFRecord (.record) file.", type=str)
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. "
                         "Defaults to the same directory as csv_DIR.",
                    type=str, default=None)

args = parser.parse_args()

if args.image_dir is None:
    args.image_dir = args.csv_dir

label_map = label_map_util.load_labelmap(args.labels_path)
label_map_dict = label_map_util.get_label_map_dict(label_map)


def txt_to_csv(path):
    objects = {
        0: 'Scalpel',
        1: 'Retractor',
        2: 'Scissors',
        3: 'BabcockForceps',
        4: 'Clamp',
        5: 'CurvedScissors',
        6: 'Forceps'
    }
    csv_list = []
    for csv_file in glob.glob(path + '/*.csv'):
        with open(csv_file, mode='r') as in_file:
            reader = csv.reader(in_file, delimiter=',')
            header_count = 0
            for row in reader:
                if header_count == 0:
                    header_count += 1
                    continue
                img_path = path + "/" + row[5] + ".jpg"
                dimens = cv2.imread(img_path)
                value = (row[5] + '.jpg',
                        int(dimens.shape[1]),
                        int(dimens.shape[0]), 
                        objects[int(float(row[0]))], 
                        int(float(row[1])), 
                        int(float(row[2])), 
                        int(float(row[3])), 
                        int(float(row[4])))
                csv_list.append(value)
    column_name = ['filename', 'width', 'height',
                  'class', 'xmin', 'ymin', 'xmax', 'ymax']
    txt_df = pd.DataFrame(csv_list, columns=column_name)
    return txt_df


def class_text_to_int(row_label):
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        print(row['xmin'], " ", row['ymin'], " ", row['xmax'], " ", row['ymax'])
        xmn = row['xmin'] / width
        if xmn < 0.0:
            xmn = 0.0
        elif xmn > 1.0:
            xmn = 1.0
        xmins.append(xmn)

        xmx = row['xmax'] / width
        if xmx < 0.0:
            xmx = 0.0
        elif xmx > 1.0:
            xmx = 1.0
        xmaxs.append(xmx)

        ymn = row['ymin'] / height
        if ymn < 0.0:
            ymn = 0.0
        elif ymn > 1.0:
            ymn = 1.0
        ymins.append(ymn)

        ymx = row['ymax'] / height
        if ymx < 0.0:
            ymx = 0.0
        elif ymx > 1.0:
            ymx = 1.0
        ymaxs.append(ymx)
        
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(args.output_path)
    path = os.path.join(args.image_dir)
    examples = txt_to_csv(args.csv_dir)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    tf.app.run()