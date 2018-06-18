#!/usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
# Copyright (c) Jonathan Dekhtiar - contact@jonathandekhtiar.eu
# All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

import time

import numpy as np
import tensorflow as tf


def my_input(data_dir, input_shape=(572, 572, 1), mask_shape=(388, 388, 1), training=True, batch_size=32, num_threads=32, seed=None):
    """
    create an input function reading a file with the Dataset API
    """

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
                size=resize_shape[:1],
                method=tf.image.ResizeMethod.BILINEAR  # [BILINEAR, NEAREST_NEIGHBOR, BICUBIC, AREA]
                align_corners=False,
                preserve_aspect_ratio=True
            )

            #if resize_shape[-1] == 1:
            #   image = tf.image.rgb_to_grayscale(image)
            # image.set_shape(resize_shape)

            return image

        input_image = decode_image(image_dir + input_image_name, resize_shape=input_shape)

        mask_image = tf.cond(
            tf.equal(image_mask_name, ""),
            true_fn=lambda: tf.zeros(mask_shape, dtype=tf.uint8),
            false_fn=lambda: decode_image(mask_image_dir + image_mask_name, resize_shape=mask_shape),
        )

        return input_image, mask_image, label

    if training:
        csv_file = os.path.join(data_dir, "train_list.csv")
        image_dir = os.path.join(data_dir, "Train")

    else:
        csv_file = os.path.join(data_dir, "test_list.csv")
        image_dir = os.path.join(data_dir, "Test")

    mask_image_dir = os.path.join(image_dir, "Labels")

    dataset = tf.data.TextLineDataset(csv_file)

    dataset = dataset.skip(1)  # Skip CSV Header

    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=decode_csv,
            num_parallel_calls=num_threads,
            batch_size=batch_size,
            drop_remainder=True,
        )
    )

    dataset = dataset.cache()

    if training:

        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
            buffer_size=shuffle_buffer_size,
            seed=seed
        ))

    else:
        dataset = dataset.repeat()

    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset


if __name__ == "__main__":

    DATA_DIR = "/data/dagm2007/Class1/"
    BURNIN_STEPS = 25
    BATCH_SIZE = 64

    # Build the data input
    dataset = my_input(
        data_dir=DATA_DIR,
        input_shape=(572, 572, 1),
        mask_shape=(388, 388, 1),
        batch_size=BATCH_SIZE,
        training=True,
        num_threads=32,
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

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        step = 0
        total_files_processed = 0

        img_per_sec_arr = []
        processing_time_arr = []

        processing_start_time = time.time()

        for _ in range(250):

            start_time = time.time()

            img_batch, mask_batch, lbl_batch = sess.run([input_images, mask_images, labels])

            print("Batch Shape: %s" % str(img_batch.shape))
            print("Mask Shape: %s" % str(mask_batch.shape))
            print("Label Shape: %s" % str(lbl_batch.shape))
            print()

            batch_size = img_batch.shape[0]
            total_files_processed += batch_size

            elapsed_time = (time.time() - start_time) * 1000
            imgs_per_sec = (batch_size / elapsed_time) * 1000

            step += 1

            if step > BURNIN_STEPS:
                processing_time_arr.append(elapsed_time)
                img_per_sec_arr.append(imgs_per_sec)

            print("\t[STEP %03d] # Files: %03d - Time: %03d msecs - Speed: %6d img/s" % (
                step,
                batch_size,
                elapsed_time,
                imgs_per_sec
            ))

        processing_time = time.time() - processing_start_time

        avg_processing_speed = np.mean(img_per_sec_arr)

        print("\n###################################################################")
        print("    *** Data Loading Performance Metrics ***\n")
        print("        => Number of Steps: %d" % step)
        print("        => Batch Size: %d" % BATCH_SIZE)
        print("        => Files Processed: %d" % total_files_processed)
        print("        => Total Execution Time: %d secs" % processing_time)
        print("        => Median Time per step: %3d msecs" % np.median(processing_time_arr))
        print("        => Median Processing Speed: %d images/secs" % np.median(img_per_sec_arr))
        print("        => Median Processing Time: %.2f msecs/image" % (1 / float(np.median(img_per_sec_arr)) * 1000))
