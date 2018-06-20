#!/usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
# Copyright (c) Jonathan Dekhtiar - contact@jonathandekhtiar.eu
# All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

import os

import tensorflow as tf

__all__ = ['data_loading_fn']


def data_loading_fn(
    data_dir,
    input_shape=(572, 572, 1),
    mask_shape=(388, 388, 1),
    training=True,
    batch_size=32,
    num_threads=32,
    use_gpu_prefetch=False,
    seed=None
):

    shuffle_buffer_size = 10000

    def decode_csv(line):

        input_image_name, image_mask_name, label = tf.decode_csv(
            line,
            record_defaults=[[""], [""], [0]],
            field_delim=','
        )

        def decode_image(filepath, resize_shape):
            image_content = tf.read_file(filepath)

            # image = tf.image.decode_image(image_content, channels=resize_shape[-1])
            image = tf.image.decode_png(
                contents=image_content,
                channels=resize_shape[-1],
                dtype=tf.uint8
            )

            image = tf.image.resize_images(
                image,
                size=resize_shape[:2],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,  # [BILINEAR, NEAREST_NEIGHBOR, BICUBIC, AREA]
                align_corners=False,
                preserve_aspect_ratio=True
            )

            image.set_shape(resize_shape)

            return image

        input_image = decode_image(
            filepath=tf.strings.join([image_dir, input_image_name], separator='/'),
            resize_shape=input_shape
        )

        mask_image = tf.cond(
            tf.equal(image_mask_name, ""),
            true_fn=lambda: tf.zeros(mask_shape, dtype=tf.uint8),
            false_fn=lambda: decode_image(tf.strings.join([mask_image_dir, image_mask_name], separator='/'), resize_shape=mask_shape),
        )

        return input_image, mask_image, label

    if training:
        csv_file = os.path.join(data_dir, "train_list.csv")
        image_dir = os.path.join(data_dir, "Train")

    else:
        csv_file = os.path.join(data_dir, "test_list.csv")
        image_dir = os.path.join(data_dir, "Test")

    mask_image_dir = os.path.join(image_dir, "Label")

    dataset = tf.data.TextLineDataset(csv_file)

    dataset = dataset.skip(1)  # Skip CSV Header

    dataset = dataset.cache()

    if training:

        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
            buffer_size=shuffle_buffer_size,
            seed=seed
        ))

    else:
        dataset = dataset.repeat()

    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=decode_csv,
            num_parallel_calls=num_threads,
            batch_size=batch_size,
            drop_remainder=True,
        )
    )

    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    if use_gpu_prefetch:
        dataset.apply(tf.data.experimental.prefetch_to_device(
            device="/gpu:0",
            buffer_size=batch_size*8
        ))

    return dataset


if __name__ == "__main__":

    '''
    Data Loading Benchmark Usage:
    python data_loading.py \    
        --data_dir="/data/dagm2007/" \
        --batch_size=64 \
        --warmup_steps=200 \
        --benchmark_steps=2000 \
        --class_id=1
    '''

    import time
    import argparse

    import numpy as np

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description="DAGM2007_data_loader_benchmark")

    parser.add_argument(
        '--data_dir',
        required=True,
        type=str,
        help="Directory path which contains the preprocessed DAGM 2007 dataset"
    )

    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        required=True,
        help="""Batch size used to measure performance."""
    )

    parser.add_argument(
        '--warmup_steps',
        default=200,
        type=int,
        required=True,
        help="""Number of steps considered as warmup and not taken into account for performance measurements."""
    )

    parser.add_argument(
        '--benchmark_steps',
        default=200,
        type=int,
        required=True,
        help="""Number of steps considered as warmup and not taken into account for performance measurements."""
    )

    parser.add_argument(
        '--class_id',
        default=1,
        choices=range(1, 11),  # between 1 and 10
        type=int,
        required=True,
        help="""Class ID used for benchmark."""
    )

    FLAGS, unknown_args = parser.parse_known_args()

    if len(unknown_args) > 0:

        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)

        raise ValueError("Invalid command line arg(s)")

    DATA_DIR = os.path.join(FLAGS.data_dir, "raw_images/private/Class1")

    BURNIN_STEPS = FLAGS.warmup_steps
    TOTAL_STEPS = FLAGS.warmup_steps + FLAGS.benchmark_steps

    BATCH_SIZE = FLAGS.batch_size

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError("The directory `%s` does not exists ..." % DATA_DIR)

    # Build the data input
    dataset = data_loading_fn(
        data_dir=DATA_DIR,
        input_shape=(572, 572, 1),
        mask_shape=(388, 388, 1),
        batch_size=BATCH_SIZE,
        training=True,
        num_threads=64,
        use_gpu_prefetch=True,
        seed=None
    )

    dataset_iterator = dataset.make_one_shot_iterator()

    input_images, mask_images, labels = dataset_iterator.get_next()

    print("Input Image Shape: %s - Dtype: %s" % (input_images.get_shape(), input_images.dtype))
    print("Mask Image Shape: %s - Dtype: %s" % (mask_images.get_shape(), mask_images.dtype))
    print("Label Shape: %s - Dtype: %s" % (labels.get_shape(), labels.dtype))

    with tf.device("/gpu:0"):
        input_images = tf.identity(input_images)
        mask_images = tf.identity(mask_images)
        labels = tf.identity(labels)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        total_files_processed = 0

        img_per_sec_arr = []
        processing_time_arr = []

        processing_start_time = time.time()

        for step in range(TOTAL_STEPS):

            start_time = time.time()

            img_batch, mask_batch, lbl_batch = sess.run([input_images, mask_images, labels])

            batch_size = img_batch.shape[0]
            total_files_processed += batch_size

            elapsed_time = (time.time() - start_time) * 1000
            imgs_per_sec = (batch_size / elapsed_time) * 1000

            if (step + 1) > BURNIN_STEPS:
                processing_time_arr.append(elapsed_time)
                img_per_sec_arr.append(imgs_per_sec)

            if (step + 1) % 20 == 0 or (step + 1) == TOTAL_STEPS:

                print("[STEP %04d] # Files: %03d - Time: %03d msecs - Speed: %6d img/s" % (
                    step + 1,
                    batch_size,
                    elapsed_time,
                    imgs_per_sec
                ))

        processing_time = time.time() - processing_start_time

        avg_processing_speed = np.mean(img_per_sec_arr)

        print("\n###################################################################")
        print("*** Data Loading Performance Metrics ***\n")
        print("\t=> Number of Steps: %d" % (step + 1))
        print("\t=> Batch Size: %d" % BATCH_SIZE)
        print("\t=> Files Processed: %d" % total_files_processed)
        print("\t=> Total Execution Time: %d secs" % processing_time)
        print("\t=> Median Time per step: %3d msecs" % np.median(processing_time_arr))
        print("\t=> Median Processing Speed: %d images/secs" % np.median(img_per_sec_arr))
        print("\t=> Median Processing Time: %.2f msecs/image" % (1 / float(np.median(img_per_sec_arr)) * 1000))

        print("\n*** Debug Shape Information:")
        print("\t[*] Batch Shape: %s" % str(img_batch.shape))
        print("\t[*] Mask Shape: %s" % str(mask_batch.shape))
        print("\t[*] Label Shape: %s" % str(lbl_batch.shape))
