#!/usr/local/bin/python2.7
""" Animal image classification with Inception. """
#
# Copyright 2017 Emilie Cassar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTICE:
# Original sample from the TensorFlow Image Recognition. This file was heavyly
# modified to meet the requirement of this project by the author.
# Orignal file available at the TensorFlow site at:
#  https://tensorflow.org/tutorials/image_recognition/
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# ==============================================================================
#
# Animal image classification with Inception.
#
# This program creates a graph from a saved GraphDef protocol buffer,
# and runs inference on an input JPEG image. It outputs human readable
# strings of the top prediction along with their probabilities.
#
# use the --help parameter for all required and optional arguments:
# python classify_image.py --help
#
# Standard call:
# python classify_image.py --with_cam cam0x --image_file file.jpeg
#
# ==============================================================================
# Change Log:
# -----------
#   Date   Description
# -------- ---------------------------------------------------------------------
# 11/09/17 Compliance to PEP-8 Python standards
# 11/09/17 Remotly downloading retrained model
# 11/10/17 Add Tinydb support
# 11/12/17 Create/Support classify python module
# 11/20/17 Tidy code to PEP-8, add argument shortcuts and dependancies
#          Added Text and PDF reports
# ==============================================================================
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import signal
from time import time
import imghdr
import configparser
import ClassifyImage.classify as classify
import ClassifyImage.reports as reports
import tensorflow as tf
#import pyexiv2
# pyexiv2 was suppressed due to lack of support under MS Windows
# Replaced by Pillow and piexif
import piexif
import cv2
from PIL import Image
from imutils import paths
try:
    from tinydb import TinyDB
except ImportError:
    LOG_TXT = ('TinyDB was not found. ' +
               'To use reports, you need to have TinyDB installed. ' +
               'Install it using pip install tinydb')
    print(LOG_TXT)
    sys.exit(1)

try:
    import rollbar
    rollbar.init('0403824544af4b2dad25758e2bd27acb')
    rollbar.report_message('Rollbar initialization completed successfully.')
except ImportError:
    LOG_TXT = ('Rollbar was not found. ' +
               'Rollbar provided automatic error reporting and needed. ' +
               'Install it using pip install rollbar')
    print(LOG_TXT)
    sys.exit(1)

def display_version():
    """ return version string """
    return classify.VERSION_STR + "\n" + classify.VERSION_COPYRIGHT

#
# main
#
# main application
# retrained_graph.pb:
#   Binary representation of the GraphDef protocol buffer.
# retrained_label_map.txt:
#   Map from synset ID to a human readable string.
#
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=too-many-nested-blocks
# pylint: disable=too-many-function-args
# pylint: disable=too-many-locals
# pylint: disable=no-member
# pylint: disable=line-too-long
# pylint: disable=pointless-string-statement
def classify_main(_):
    """ main application """
    global FLAGS
    imagepath = ''
    # Setup TinyDB
    tinydb = TinyDB('hornaday_db.json', default_table='imagetable')
    animaltable = tinydb.table('animaltable')
    # Setup signal handler
    signal.signal(signal.SIGINT, classify.signal_handler)

    # Configuration files
    configfile = configparser.ConfigParser()
    cameraconfigfile = configparser.ConfigParser()
    #configfile._interpolation = ConfigParser.ExtendedInterpolation()
    configfile.read(FLAGS.config_file)
    cameraconfigfile.read(FLAGS.config_camera_file)

    # Reports
    textreport = False
    pdfreport = False
    testdatareport = False
    if FLAGS.text_report:
        textreport = True
    if FLAGS.pdf_report:
        pdfreport = True
    if FLAGS.with_test_data:
        testdatareport = True
    if (textreport or pdfreport):
        reports.run_report(textreport, pdfreport, testdatareport)
        sys.exit(0)
    # Display Version and Author
    if FLAGS.version:
        print(display_version())
        sys.exit(0)
    # Setup logging
    if FLAGS.log:
        loglevel = FLAGS.log.upper()
        logger = classify.create_logger("IMG_CLASSIFIER", loglevel)

    #
    # List all cameras
    #
    if FLAGS.list_cameras:
        classify.list_cameras_definition(cameraconfigfile, FLAGS.config_camera_file, logger)

    # Check for update
    classify.update_check(FLAGS.model_dir, logger)

    # If not available, download imagenet
    #
    classify.maybe_download_and_extract(FLAGS.model_dir, logger)

    # If not available, download retrained imagenet
    #
    classify.maybe_dwnld_and_extract_model(FLAGS.model_dir, logger)
    result = []
    #
    # Process an image directory as specified under command line
    # parameter --image_dir
    #
    print(FLAGS.image_dir)
    if FLAGS.image_dir:
        input_dir = os.path.abspath(FLAGS.image_dir)
        imagepaths = list(paths.list_images(input_dir))
        # pylint: disable=unused-variable
        for (i, imagepath) in enumerate(imagepaths):
            imagepath = imagepath.replace('\\', '')
            if imghdr.what(imagepath) is not None:
                if FLAGS.with_cam:
                    if classify.check_image_cam_db(animaltable, imagepath, FLAGS.with_cam):
                        logtxt = ("Image " +
                                  imagepath +
                                  " on camera id " +
                                  FLAGS.with_cam +
                                  " already processed. Skipping image.")
                        logger.info(logtxt)
                        print(logtxt)
                        continue
                elif classify.check_image_db(animaltable, imagepath):
                    logtxt = "Image " + imagepath + " already processed. Skipping image."
                    logger.info(logtxt)
                    print(logtxt)
                    continue
                start_time = time()
                #print("Processing image file " + imagepath)
                # pylint: disable=not-context-manager
                with tf.Graph().as_default():
                    result = classify.run_interference_on_image(imagepath, FLAGS.num_top_predictions)
                if result[1] * 100 >= 30:
                    logtxt = ("Image: " + imagepath + " will be classify as [" +
                              str(result[0].rstrip(",")) +
                              "] Score: {:.2f}% - {:.2f} seconds"
                              .format(result[1] * 100, time() - start_time))
                    print(logtxt)
                    logger.info(logtxt)
                    animalscore = str(result[0].rstrip(","))
                    if FLAGS.save_image:
                        input_img = cv2.imread(imagepath)
                        cv2.putText(input_img,
                                    str(result[0].rstrip(",") + " - {:.2f}%"
                                        .format(result[1] * 100)), (10, 30),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 1, 8, False)
                        # pylint: enable=bare-except
                        # Save the image
                        # Get Image creation date from Exif metadata
                        # Key: Image DateTime, value 2017:06:15 07:12:53
                        image_file = Image.open(r"" + imagepath)
                        try:
                            exif_dict = piexif.load(image_file.info["exif"])
                            #pylint: disable=maybe-no-member
                            dtoriginal = classify.exif_decode_encode(
                                exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal])
                            datetimeoriginal = ''.join(dtoriginal.encode("utf-8"))
                            #pylint: enable=maybe-no-member
                            image_file.close()
                            datetemp, timetemp = datetimeoriginal.split(' ')
                            dateyear, datemonth, dateday = datetemp.split(':')
                            # Now create supporting directories
                            dest_directory = ''.join([FLAGS.classified_image_dir,
                                                      '/', dateyear, '/', datemonth, '/', dateday, '/',
                                                      str(result[0].rstrip(","))])
                            if not os.path.exists(dest_directory):
                                os.makedirs(dest_directory)
                            save_image_path = (dest_directory + "/" + os.path.basename(imagepath))
                            # pylint: disable=no-member
                            cv2.imwrite(save_image_path, input_img)
                            # pylint: enable=no-member
                            classify.set_author_image(save_image_path, logger)
                            docid = classify.insert_update_db(
                                animaltable,
                                animalscore,
                                imagepath,
                                datetemp,
                                timetemp,
                                result[1] * 100)
                            if FLAGS.with_cam:
                                classify.geo_encode_image(
                                    save_image_path,
                                    FLAGS.with_cam,
                                    cameraconfigfile,
                                    logger)
                                classify.add_set_geo_db(
                                    animaltable,
                                    save_image_path,
                                    FLAGS.with_cam,
                                    docid,
                                    cameraconfigfile)
                        except KeyError:
                            logtxt = "KE1 - KeyError received for image " + imagepath + ". Skipping image."
                            logger.error(logtxt)
                            print(logtxt)
                    else:
                        if FLAGS.with_cam:
                            classify.geo_encode_image(imagepath, FLAGS.with_cam, cameraconfigfile, logger)
                else:
                    logtxt = ("Image: " + imagepath +
                              " not classified. Prediction to low. Score: {:.2f}% - {:.2f} seconds"
                              .format(result[1] * 100, time() - start_time))
                    print(logtxt)
                    logger.warning(logtxt)
                    if FLAGS.save_image:
                        # Copy this image to the not_classified folder
                        input_img = cv2.imread(imagepath)
                        image_file = Image.open(r"" + imagepath)
                        try:
                            exif_dict = piexif.load(image_file.info["exif"])
                            #pylint: disable=maybe-no-member
                            datetimeoriginal = classify.exif_decode_encode(
                                exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal])
                            #pylint: enable=maybe-no-member
                            image_file.close()
                            datetemp, timetemp = datetimeoriginal.split(' ')
                            dateyear, datemonth, dateday = datetemp.split(':')
                            # Now create supporting directories
                            dest_directory = ''.join([FLAGS.classified_image_dir,
                                                      '/', dateyear, '/', datemonth, '/', dateday,
                                                      "/not_classified"])
                            if not os.path.exists(dest_directory):
                                os.makedirs(dest_directory)
                            # Save the image
                            save_image_path = (dest_directory + "/" + os.path.basename(imagepath))
                            # pylint: disable=no-member
                            cv2.imwrite(save_image_path, input_img)
                            # pylint: enable=no-member
                            classify.set_author_image(save_image_path, logger)
                            docid = classify.insert_update_db(
                                animaltable,
                                'not_classified',
                                imagepath,
                                datetemp,
                                timetemp,
                                result[1] * 100)
                            if FLAGS.with_cam:
                                classify.geo_encode_image(
                                    save_image_path,
                                    FLAGS.with_cam,
                                    cameraconfigfile,
                                    logger)
                                classify.add_set_geo_db(
                                    animaltable,
                                    save_image_path,
                                    FLAGS.with_cam,
                                    docid,
                                    cameraconfigfile)
                        except KeyError:
                            logtxt = "KE2 - KeyError received for image " + imagepath + "."
                            logger.error(logtxt)
                            print(logtxt)
                # pylint: enable=not-context-manager
        # pylint: enable=unused-variable
    # Process a unique image as specified under command line
    # parameter ==image_file
    if FLAGS.image_file:
        start_time = time()
        # pylint: disable=not-context-manager
        with tf.Graph().as_default():
            result = classify.run_interference_on_image(FLAGS.image_file, FLAGS.num_top_predictions)
        if result[1] * 100 >= 30:
            logtxt = ("Image: " + FLAGS.image_file + " will be classify as [" +
                      str(result[0].rstrip(",")) +
                      "] Score: {:.2f}% - {:.2f} seconds"
                      .format(result[1] * 100, time() - start_time))
            print(logtxt)
            logger.info(logtxt)
            animalmatch = str(result[0].rstrip(","))
        else:
            logtxt = ("Image: " + FLAGS.image_file +
                      " not classified. Prediction to low. Score: {:.2f}% - {:.2f} seconds"
                      .format(result[1] * 100, time() - start_time))
            print(logtxt)
            logger.warning(logtxt)
            animalmatch = "not_classified"
        # Geo exif encoding for this image
        if FLAGS.with_cam:
            if FLAGS.save_image:
                if FLAGS.classified_image_dir:
                    input_img = cv2.imread(FLAGS.image_file)
                    image_file = Image.open(r"" + FLAGS.image_file)
                    try:
                        exif_dict = piexif.load(image_file.info["exif"])
                        #pylint: disable=maybe-no-member
                        datetimeoriginal = classify.exif_decode_encode(
                            exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal])
                        #pylint: enable=maybe-no-member
                        datetemp, timetemp = datetimeoriginal.split(' ')
                        dateyear, datemonth, dateday = datetemp.split(':')
                        # Now create supporting directories
                        dest_directory = ''.join([FLAGS.classified_image_dir,
                                                  '/', dateyear, '/', datemonth, '/', dateday, '/',
                                                  str(result[0].rstrip(","))])
                        if not os.path.exists(dest_directory):
                            os.makedirs(dest_directory)
                        save_image_path = (dest_directory + "/" + os.path.basename(FLAGS.image_file))
                        # pylint: disable=no-member
                        cv2.imwrite(save_image_path, input_img)
                        # pylint: enable=no-member
                        docid = classify.insert_update_db(
                            animaltable,
                            animalmatch,
                            FLAGS.image_file,
                            datetemp,
                            timetemp,
                            result[1] * 100)
                        classify.set_author_image(save_image_path, logger)
                        classify.geo_encode_image(save_image_path, FLAGS.with_cam, cameraconfigfile, logger)
                    except KeyError:
                        logtxt = "KE3 - KeyError received for image " + imagepath + "."
                        logger.error(logtxt)
                        print(logtxt)
                else:
                    logtxt = "--save_image requires --classified_image_dir argument"
                    logger.warning(logtxt)
                    print(logtxt)
            else:
                logtxt = "--with_cam requires --save_image argument"
                logger.warning(logtxt)
                print(logtxt)
        # Display exif metadata for this image
        if FLAGS.with_exif:
            classify.list_exif_image(FLAGS.image_file)
        if FLAGS.show_image:
            # Let timestamp the image
            input_img = cv2.imread(FLAGS.image_file)
            cv2.putText(input_img,
                        str(result[0].rstrip(",") + " - {:.2f}%"
                            .format(result[1] * 100)), (5, 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1, 8, False)
            cv2.imshow("{} - {:.2f}%".format(result[0].rstrip(","), result[1] * 100), input_img)
            cv2.waitKey(0)
        #pylint: enable=not-context-manager

"""
usage: classify_image.py [-h] [-v] [-md MODEL_DIR] [-if IMAGE_FILE]
                         [-cf CONFIG_FILE] [-lc] [-wc WITH_CAM]
                         [-id IMAGE_DIR] [-ntp NUM_TOP_PREDICTIONS] [-we]
                         [-di] [-si] [-cid CLASSIFIED_IMAGE_DIR] [-l LOG]

classify_image.py Python Script version 11.12.17 Copyright (c) Emilie Cassar,
2017

optional arguments:
  -h, --help            show this help message and exit

General:
  General options

  -v, --version         Display classify_image version and author information.
  -cf CONFIG_FILE, --config_file CONFIG_FILE
                        Full path and filename to configuration file.
  -lc, --list_cameras   List defined cameras in configuration file and their
                        detail, then exit.
  -l LOG, --log LOG     Set logging level. Valid values are INFO, WARNING,
                        ERROR and DEBUG.

Image:
  Image processing options

  -md MODEL_DIR, --model_dir MODEL_DIR
                        Path to classify_image_graph_def.pb,
                        imagenet_synset_to_human_label_map.txt, and
                        imagenet_2012_challenge_label_map_proto.pbtxt.
  -if IMAGE_FILE, --image_file IMAGE_FILE
                        Absolute path to image file.
  -wc WITH_CAM, --with_cam WITH_CAM
                        Configured camera to use for image(s).
  -id IMAGE_DIR, --image_dir IMAGE_DIR
                        Absolute path to image directory.
  -ntp NUM_TOP_PREDICTIONS, --num_top_predictions NUM_TOP_PREDICTIONS
                        Display this many predictions.
  -we, --with_exif      Display exif data for a single image if available.
  -di, --show_image     Display the single image with prediction and encoding.
  -si, --save_image     Save the generated classified image with text overlay
                        and exif metadata.
  -cid CLASSIFIED_IMAGE_DIR, --classified_image_dir CLASSIFIED_IMAGE_DIR
                        Absolute path to the classified image directory.

Report:
  Reporting options
"""

def classify_run(args):
    ''' Run classify application '''
    global FLAGS
    PARSER = classify.ArgParser(
        description=display_version(),
        prog=args[0],
        add_help=True)
    # pylint: disable=invalid-name
    group1 = PARSER.add_argument_group('General', 'General options')
    group2 = PARSER.add_argument_group('Image', 'Image processing options')
    group3 = PARSER.add_argument_group('Report', 'Reporting options')
    group4 = group2.add_mutually_exclusive_group()
    # pylint: enable=invalid-name
    group1.add_argument(
        '-v',
        '--version',
        action='store_true',
        default=False,
        help='Display classify_image version and author information.'
    )
    group2.add_argument(
        '-md',
        '--model_dir',
        type=str,
        default='./',
        help="""\
        Path to classify_image_graph_def.pb,
        imagenet_synset_to_human_label_map.txt, and
        imagenet_2012_challenge_label_map_proto.pbtxt.\
        """
    )
    group4.add_argument(
        '-if',
        '--image_file',
        type=str,
        default='',
        help='Absolute path to image file.'
    )
    group1.add_argument(
        '-ccf',
        '--config_camera_file',
        type=str,
        default='./camera.ini',
        help='Full path and filename to camera configuration file.'
    )
    group1.add_argument(
        '-cf',
        '--config_file',
        type=str,
        default='./config.ini',
        help='Full path and filename to configuration file.'
    )
    #PARSER.add_argument(
    #    '--ui',
    #    action='store_true',
    #    default=False,
    #    help='Display classify_image User Interface'
    #)
    group1.add_argument(
        '-lc',
        '--list_cameras',
        action='store_true',
        default=False,
        help='List defined cameras in configuration file and their detail, then exit.'
    )
    group2.add_argument(
        '-wc',
        '--with_cam',
        type=str,
        action=classify.SaveImageAction,
        default='',
        help='Configured camera to use for image(s).'
    )
    group4.add_argument(
        '-id',
        '--image_dir',
        type=str,
        default='',
        help='Absolute path to image directory.'
    )
    group2.add_argument(
        '-ntp',
        '--num_top_predictions',
        type=int,
        default=1,
        help='Display this many predictions.'
    )
    group2.add_argument(
        '-we',
        '--with_exif',
        action='store_true',
        default=False,
        help='Display exif data for a single image if available.'
    )
    group2.add_argument(
        '-di',
        '--show_image',
        action='store_true',
        default=False,
        help='Display the single image with prediction and encoding.'
    )
    group2.add_argument(
        '-si',
        '--save_image',
        action='store_true',
        default=False,
        help='Save the generated classified image with text overlay and exif metadata.'
    )
    group2.add_argument(
        '-cid',
        '--classified_image_dir',
        type=str,
        default='./classified_images',
        help='Absolute path to the classified image directory.'
    )
    group1.add_argument(
        '-l',
        '--log',
        type=str,
        default="INFO",
        help='Set logging level. Valid values are INFO, WARNING, ERROR and DEBUG.'
    )
    group3.add_argument(
        '-tr',
        '--text_report',
        action='store_true',
        default=False,
        help='Generate text table report based on database data, display report to standard output (console).'
    )
    group3.add_argument(
        '-pr',
        '--pdf_report',
        action='store_true',
        default=False,
        help='Generate PDF report based on database data, save report to file.'
    )
    group3.add_argument(
        '-wtd',
        '--with_test_data',
        action='store_true',
        default=False,
        help='Generate text or PDF report based on test data not database data (Testing).'
    )
    prog = args[0]
    args.pop(0)
    FLAGS = PARSER.parse_args(args)
    tf.app.run(main=classify_main, argv=[prog])
