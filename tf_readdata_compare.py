# coding: utf-8

import shutil
import sys 
import tensorflow as tf
import datetime
import numpy as np

def make_tld(csv_filename, header_lines, delim, batch_size):
    dataset = tf.data.TextLineDataset(filenames=csv_filename).skip(header_lines)
    num_cols_ = 114
    def parse_csv(line):
        cols_types = [[]] * num_cols_  # all required
        columns = tf.decode_csv(line, record_defaults=cols_types, field_delim=delim)
        return tf.stack(columns)

    dataset = dataset.map(parse_csv).batch(batch_size)
    return dataset


def make_tsd(csv_filename, header_lines, delim, batch_size):
    with open(csv_filename, "r") as f:
        lines = f.readlines()

    data_shape = (len(lines) - header_lines, len(lines[header_lines].strip().split(delim)))
    data = np.empty(shape=data_shape, dtype=np.float32)

    for idx, line in enumerate(lines[header_lines:]):
        columns = [float(el) for el in line.strip().split(delim)]
        data[idx, :] = np.array(columns)

    dataset = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
    return dataset


if __name__ == "__main__":
    batch_size_ = 100
    csv_filename_ = "/data/chenmingming/push_wide_deep/2018-05-03/deep.test.3"
    delim_ = " "
    header_lines_ = 0
    tld_start = datetime.datetime.now()
    tld = make_tld(csv_filename_, header_lines_, delim_, batch_size_)
    tld_next = tld.make_one_shot_iterator().get_next()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    with tf.Session() as tld_sess:
        tld_sess.run(tf.global_variables_initializer())
        try:
            while True:
                tld_out = tld_sess.run(tld_next)
        except tf.errors.OutOfRangeError:
            print("Done")
    tld_end = datetime.datetime.now()
    print("TextLineDataset: " + str(tld_end - tld_start))

    tsd_start = datetime.datetime.now()
    tsd = make_tsd(csv_filename_, header_lines_, delim_, batch_size_)
    tsd_next = tsd.make_one_shot_iterator().get_next()
    with tf.Session() as tsd_sess:
        tsd_sess.run(tf.global_variables_initializer())
        try:
            while True:
                tsd_out = tsd_sess.run(tsd_next)
        except tf.errors.OutOfRangeError:
            print("Done")
    tsd_end = datetime.datetime.now()
    print("TensorSliceDataset: " + str(tsd_end - tsd_start))

"""
the time used with different reading methods are quite different,

reading file of 200000 * 114 float matrix data with tf 1.6
1. TextLineDataset: 0:01:41.747109
2. TensorSliceDataset: 0:00:12.382953


reading file of 2000000 * 114 float matrix data with tf 1.6
1. TextLineDataset: 0:16:20.066501
2. TensorSliceDataset: 0:02:04.559696

"""
