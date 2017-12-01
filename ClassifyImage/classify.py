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
# 11/10/17 Add tinydb support
#
# ==============================================================================
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import signal
from time import time
import tarfile
import math
import imghdr
import configparser
import logging
import numpy as np
from six.moves import urllib
import tensorflow as tf
#import pyexiv2
# pyexiv2 was suppressed due to lack of support under MS Windows
# Replaced by Pillow and piexif
import piexif
import cv2
import exifread
from PIL import Image
from imutils import paths
try:
    from tinydb import TinyDB, Query
    from tinydb.operations import add, set
except ImportError:
    LOG_TXT = ('TinyDB was not found. ' +
               'To use reports, you need to have TinyDB installed. ' +
               'Install it using pip install tinydb')
    print(LOG_TXT)
    sys.exit(1)

try:
    import rollbar
except ImportError:
    LOG_TXT = ('Rollbar was not found. ' +
               'Rollbar provided automatic error reporting and needed. ' +
               'Install it using pip install rollbar')
    print(LOG_TXT)
    sys.exit(1)

# Suppress tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = None

# pylint: disable=no-member
# pylint: disable=old-style-class
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-lines
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=redefined-builtin
# pylint: disable=line-too-long
FILE_WS = 'http://www.cassartx.net'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
RETRAINED_URL = FILE_WS + '/hornaday/retrained-latest.tgz'
VERSION_URL = FILE_WS + '/hornaday/remote_version.txt'
SCRIPT_URL = FILE_WS + '/hornaday/classify-latest.tgz'
LEGAL_WS = 'https://www.cassartx.us'
# pylint: enable=line-too-long
VERSION = '12.01.17'
VERSION_STR = "classify_image.py Python Script version " + VERSION
VERSION_COPYRIGHT = "Copyright (c) Emilie Cassar, 2017"
# Loads label file, strips off carriage return
LABEL_LINES = [line.rstrip() for line
               in tf.gfile.GFile("retrained_labels.txt")]

class ArgParser(argparse.ArgumentParser):
    """ ArgParser - Standard argument parser class """

    def error(self, message):
        """ error - Args: message -- The error message. Exit with return code 2 """
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

class SaveImageAction(argparse.Action):
    """ Check if argument is specify on command line """
    # pylint: disable=too-few-public-methods
    def __call__(self, parser, namespace, values, option_string=None):
        if not namespace.save_image:
            parser.error('Missing save_image')
        else:
            namespace.with_cam.append(values)

# pylint: disable=no-member
# pylint: disable=old-style-class
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-lines
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=redefined-builtin
class Rational:
    """ A simple fraction class. Python 2.6 could use the inbuilt Fraction class. """

    def __init__(self, num, den):
        """ Create a number fraction num/den. """
        self.num = num
        self.den = den

    def __repr__(self):
        """ Return a string representation of the fraction. """
        return "%s / %s" % (self.num, self.den)

    def as_tuple(self):
        """ Return the fraction a numerator, denominator tuple. """
        return (self.num, self.den)

# pylint: disable=unused-argument
def signal_handler(sig, frame):
    """ signal_handler - Handle keyboard interruption """
    print('You pressed Ctrl+C!')
    sys.exit(0)
# pylint: enable=unused-argument
# pylint: enable=old-style-class
# pylint: enable=too-few-public-methods

#
# create_logger
#
# Create a logging interface
#
#  Args:
#    app_name -- The application name
#    log_kevel -- The logging level
#
#  Returns:
#    logger -- the logging interface object
#
def create_logger(app_name, log_level):
    """ Create a logging interface """
    logname = os.path.splitext(__file__)[0] + ".log"
    logging.basicConfig(
        filename=logname,
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(app_name)
    return logger

#
# to_deg
#
# Convert a location to degrees, minutes, seconds
#
#  Args
#    value -- The float number to be converted
#    loc -- A list of location value
#
#  Returns
#    list object with deg, minute, sec, loc value
#
def to_deg(value, loc):
    """ Convert a location to degrees, minutes, seconds """
    if value < 0:
        loc_value = loc[0]
    elif value > 0:
        loc_value = loc[1]
    else:
        loc_value = ""
    abs_value = abs(value)
    deg = int(abs_value)
    tmp = (abs_value-deg)*60
    minute = int(tmp)
    sec = round((tmp - minute)* 60, 5)
    return (deg, minute, sec, loc_value)

#
# to_dms
#
# Convert a decimal value location to degrees, minutes, seconds
#
#  Args
#    val -- The float location to be converted
#
#  Returns
#    list sign value, degrees, minutes, seconds
#
def to_dms(val):
    """ Convert a decimal value location to degrees, minutes, seconds """
    secden = 50000000
    sign = 1
    if val < 0:
        val = -val
        sign = -1

    deg = int(val)
    other = (val - deg) * 60
    minutes = int(other)
    secs = (other - minutes) * 60
    secs = long(secs * secden)
    return (sign, deg, minutes, secs)

#
# set_gps_location
#
# Adds GPS position as EXIF metadata
#   Args:
#     file_name -- image file
#     lat -- latitude (as float)
#     lng -- longitude (as float)
#     logger -- the logging interface object
#     cam -- the camera id
#
#   Returns:
#     None
#
def set_gps_location(file_name, lat, lng, logger, cam):
    """ Adds GPS position as EXIF metadata """
    lat_deg = to_deg(lat, ["S", "N"])
    lng_deg = to_deg(lng, ["W", "E"])

    logger.debug(lat_deg)
    logger.debug(lng_deg)

    # convert decimal coordinates into degrees, minutes and seconds
    exiv_lat = (Rational(int(math.floor(lat_deg[0]*60+lat_deg[1])), 60).as_tuple(),
                Rational(int(math.floor(lat_deg[2]*100)), 6000).as_tuple(),
                Rational(0, 1).as_tuple())
    exiv_lng = (Rational(int(math.floor(lng_deg[0]*60+lng_deg[1])), 60).as_tuple(),
                Rational(int(math.floor(lng_deg[2]*100)), 6000).as_tuple(),
                Rational(0, 1).as_tuple())
    image_file = Image.open(r"" + file_name)
    #pylint: disable=maybe-no-member
    exif_dict = piexif.load(file_name)
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = cam
    exif_dict["0th"][piexif.ImageIFD.GPSTag] = 654
    exif_dict["GPS"][piexif.GPSIFD.GPSMapDatum] = "WGS-84"
    exif_dict["GPS"][piexif.GPSIFD.GPSVersionID] = (2, 0, 0, 0)
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = lat_deg[3]
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = exiv_lat
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = lng_deg[3]
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = exiv_lng
    #pylint: enable=maybe-no-member

    exif_bytes = piexif.dump(exif_dict)
    image_file.save(file_name, "jpeg", exif=exif_bytes)
    image_file.close()

#
# set_author_image
#
# Adds Authoring EXIF metadata to image
#   Args:
#     file_name -- image file
#     logger -- the logging interface object
#
#   Returns:
#     None
#
def set_author_image(file_name, logger):
    """ Adds Authoring EXIF metadata to image """
    logger.info("Setting authoring Exif metadata on file " + file_name)
    image_file = Image.open(r"" + file_name)
    #exif_dict = piexif.load(image_file.info["exif"])
    exif_dict = piexif.load(file_name)
    #pylint: disable=maybe-no-member
    exif_dict["0th"][piexif.ImageIFD.Artist] = "EJC Hornaday Project"
    exif_dict["0th"][piexif.ImageIFD.Copyright] = "Boy Scouts of America"
    #pylint: enable=maybe-no-member
    exif_bytes = piexif.dump(exif_dict)
    image_file.save(file_name, "jpeg", exif=exif_bytes)
    image_file.close()

#
# create_graph
#
# Creates a graph from saved GraphDef file and returns a saver.
#
#   Args:
#     None
#   Returns:
#     None
#
def create_graph():
    """ Creates graph from saved graph_def.pb. """
    # with tf.gfile.FastGFile(os.path.join(
    #    FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as graph_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph_file.read())
        _ = tf.import_graph_def(graph_def, name='')

#
# run_interference_on_image
#
# Runs inference on an image.
#
#  Args:
#    image: Image file name.
#    logger -- the logging interface object
#
#  Returns:
#    category and score as list
#
def run_interference_on_image(image, num_top_predictions):
    """ Runs inference on an image. """
    result = []
    # Check if image exist
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()
    # Creates graph from save GraphDef.
    create_graph()
    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
        # node_lookup = NodeLookup()

        top_k = predictions.argsort()[-num_top_predictions:][::-1]
        # We only need the top score
        node_id = top_k[0]
        # Loads label file, strips off carriage return
        labellines = [lineread.rstrip() for lineread
                      in tf.gfile.GFile("retrained_labels.txt")]
        human_string = labellines[node_id]
        # human_string = node_lookup.id_to_string(node_id)
        # Dictionary for us Texan.
        human_string = human_string.replace("gazelle", "deer")
        score = predictions[node_id]
        # print('%s (score = %.5f)' % (human_string, score))
        result.append(human_string)
        result.append(score)
    return result

#
# maybe_download_and_extract
#
# Download and extract model tar file.
#
#  Args:
#    logger -- the logging interface object
#
#  Returns:
#    None
#
def maybe_download_and_extract(model_dir, logger):
    """ Download and extract model tar file. """
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            done = int(50 * float(count * block_size) / total_size)
            sys.stdout.write('\r>> Downloading %s [%s%s] %.1f%%' % (
                filename, '=' * done, ' ' * (50 - done),
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        try:
            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        except IOError:
            rollbar.report_message('IOError received when trying to download ' + DATA_URL + ' in maybe_download_and_extract')
        except:
            # catch-all
            rollbar.report_exc_info()
        print()
        statinfo = os.stat(filepath)
        logtxt = ('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        print(logtxt)
        logger.info(logtxt)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

#
# maybe_dwnld_and_extract_model
#
# Download and extract retrained model tar file.
#
#  Args:
#    logger -- the logging interface object
#
#  Returns:
#    None
#
def maybe_dwnld_and_extract_model(model_dir, logger):
    """ Download and extract retrained model tar file. """
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = RETRAINED_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            done = int(50 * float(count * block_size) / total_size)
            sys.stdout.write('\r>> Downloading %s [%s%s] %.1f%%' % (
                filename, '=' * done, ' ' * (50 - done),
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        try:
            filepath, _ = urllib.request.urlretrieve(RETRAINED_URL, filepath, _progress)
        except IOError:
            rollbar.report_message('IOError received when trying to download ' + RETRAINED_URL + ' in maybe_dwnld_and_extract_model')
        except:
            # catch-all
            rollbar.report_exc_info()
        print()
        statinfo = os.stat(filepath)
        logtxt = ('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        print(logtxt)
        logger.info(logtxt)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

#
# download_latest_model
#
# Download and extract latest retrained model tar file.
#
#  Args:
#    logger -- the logging interface object
#
#  Returns:
#    None
#
def download_latest_model(model_dir, logger):
    """ Download and extract latest retrained model tar file. """
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = RETRAINED_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    def _progress(count, block_size, total_size):
        done = int(50 * float(count * block_size) / total_size)
        sys.stdout.write('\r>> Downloading %s [%s%s] %.1f%%' % (
            filename, '=' * done, ' ' * (50 - done),
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    try:
        filepath, _ = urllib.request.urlretrieve(RETRAINED_URL, filepath, _progress)
    except IOError:
        rollbar.report_message('IOError received when trying to download ' + RETRAINED_URL + ' in download_latest_model')
    except:
        # catch-all
        rollbar.report_exc_info()
    print()
    statinfo = os.stat(filepath)
    logtxt = ('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    print(logtxt)
    logger.info(logtxt)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

#
# download_latest_script
#
# Download and extract latest script tar file.
#
#  Args:
#    logger -- the logging interface object
#
#  Returns:
#    None
#
def download_latest_script(model_dir, logger):
    """ Download and extract latest script tar file. """
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = SCRIPT_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    def _progress(count, block_size, total_size):
        done = int(50 * float(count * block_size) / total_size)
        sys.stdout.write('\r>> Downloading %s [%s%s] %.1f%%' % (
            filename, '=' * done, ' ' * (50 - done),
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    try:
        filepath, _ = urllib.request.urlretrieve(SCRIPT_URL, filepath, _progress)
    except IOError:
        rollbar.report_message('IOError received when trying to download ' + SCRIPT_URL + ' in download_latest_script')
    except:
        # catch-all
        rollbar.report_exc_info()
    print()
    statinfo = os.stat(filepath)
    logtxt = ('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    print(logtxt)
    logger.info(logtxt)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

# pylint: disable=unused-argument
#
# update_check
#
# Verify retrained model and python script version and
# display message if newest available
#
def update_check(model_dir, logger):
    ''' Version checker '''

    #Gets downloaded version
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = 'version.txt'
    filepath = os.path.join(dest_directory, filename)
    remotefilename = 'remote_version.txt'
    remotefilepath = os.path.join(dest_directory, remotefilename)
    try:
        versionsource = open(filepath, 'r')
    except IOError:
        # Version is not found on local drive, so force an update
        logtext = "Version missing from local drive. Downloading latest release."
        logger.warning(logtext)
        print(logtext)
        download_latest_script(model_dir, logger)
        # Rerun the script
        os.system("python classify_image.py " + ' '.join(sys.argv[1:]))
        sys.exit(0)
    versioncontents = versionsource.read()

    #gets newest version
    _ = urllib.request.urlretrieve(VERSION_URL, remotefilepath)
    updatesource = open(remotefilepath, 'r')
    updatecontents = updatesource.read()
    # pylint: disable=line-too-long
    # checks for updates
    for i in range(0, 25):
        if updatecontents[i] != versioncontents[i]:
            logtext = "There are data updates available for the retrained ImageNet model."
            logger.warning(logtext)
            print(logtext)
            download_latest_model(model_dir, logger)
            break
    for i in range(26, 51):
        if updatecontents[i] != versioncontents[i]:
            logtext = "There are version updates available for the classify_image python script."
            logger.warning(logtext)
            print(logtext)
            download_latest_script(model_dir, logger)
            # Rerun the script
            os.system("python classify_image.py " + ' '.join(sys.argv[1:]))
            sys.exit(0)
            break
    # pylint: enable=line-too-long
# pylint: enable=unused-argument

#
# list_cameras_definition
#
# List all cameras defined in the configuration file
# camera.ini and their definition.
#
#   Args:
#     configfile: The config parser object to use
#     logger -- the logging interface object
#
#   Returns:
#     None
#
def list_cameras_definition(configfile, config_file, logger):
    """ List all cameras defined in the configuration file """
    nb_cameras = len(configfile.sections())
    logtxt = ("{} cameras found in configuration file {}"
              .format(nb_cameras, config_file))
    print(logtxt + "\n")
    logger.info(logtxt)
    # pylint: disable=bare-except
    for camera in configfile.sections():
        logtxt = ("Camera {} definition:".format(camera))
        print(logtxt)
        logger.info(logtxt)
        options = configfile.options(camera)
        for option in options:
            try:
                config_option = configfile.get(camera, option)
                logtxt = ("{}: {}".format(option, config_option))
                print(logtxt)
                logger.info(logtxt)
            except:
                logtxt = ("Exception on {}".format(option))
                print(logtxt)
                logger.error(logtxt)
                rollbar.report_exc_info()
        print("\n")
    # pylint: enable=bare-except
    sys.exit(0)

#
# geo_encode_image
#
# Add Geo information to image
#
#  Args:
#    img: Image file name
#    cam: The camera id to use for encoding exif metadata
#    configfile: The config parser object to use
#    logger -- the logging interface object
#
#  Returns:
#    None
#
def geo_encode_image(img, cam, configfile, logger):
    """ Add Geo information to image """
    # pylint: disable=bare-except
    try:
        gpslocation = configfile.get(cam, 'geo')
        camera_description = configfile.get(cam, 'Description')
        gpslongitude, gpslatitude = gpslocation.split(",")
        logtxt = ("Setting image exif GPS metadata with Longitude {} Latitude {}"
                  .format(gpslongitude.strip(), gpslatitude.strip()))
        print(logtxt)
        logger.info(logtxt)
        set_gps_location(img,
                         float(gpslongitude.strip()),
                         float(gpslatitude.strip()),
                         logger,
                         camera_description)
    except:
        logtxt = ("Geo configuration not defined for camera " + cam)
        print(logtxt)
        logger.warning(logtxt)
    # pylint: enable=bare-except

def exif_decode_encode(exifmeta):
    ''' Check if String or Bytes and encode/decode accordly '''
    exifstring = ''
    if isinstance(exifmeta, str):
        exifstring = exifmeta.encode("utf-8")
    elif isinstance(exifmeta, bytes):
        exifstring = exifmeta.decode("utf-8")
    return exifstring
#
# list_exif_image
#
# list exif metadata for an image
#
#   Args:
#     img: Image file name
#
#   Returns:
#     None
#
def list_exif_image(img):
    """ list exif metadata for an image """
    # pylint: disable=consider-iterating-dictionary
    img_f = open(img, 'rb')
    tags = exifread.process_file(img_f)
    for tag in tags.keys():
        if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
            print("Key: {}, value {}".format(tag, tags[tag]))
    img_f.close()
    #pylint: enable=consider-iterating-dictionary

def insert_update_db(animaltable, animalscore, imagepath, datetemp, timetemp, score):
    """ Update/Insert row in table """
    Animal = Query()
    if animaltable.search(Animal.image == os.path.basename(imagepath)):
        docid = animaltable.update({
            'date': datetemp,
            'time': timetemp,
            'animal': animalscore,
            'score': score,
            'image': os.path.basename(imagepath)
            }, Animal.image == os.path.basename(imagepath))
    else:
        docid = animaltable.insert({
            'date': datetemp,
            'time': timetemp,
            'animal': animalscore,
            'score': score,
            'image': os.path.basename(imagepath)
            })
    return docid

def check_image_db(animaltable, imagepath):
    """ Check if this image is in the DB, then return true if yes, false if no """
    Animal = Query()
    found = False
    if animaltable.search(Animal.image == os.path.basename(imagepath)):
        found = True
    return found

def check_image_cam_db(animaltable, imagepath, with_cam):
    """ Check if this image from this camera is in the DB, then return true if yes, false if no """
    Animal = Query()
    found = False
    if animaltable.search(
            (Animal.image == os.path.basename(imagepath)) & (Animal.camid == with_cam)):
        found = True
    return found

def add_set_geo_db(animaltable, imagepath, cam, docid, configfile):
    """ Add/Set key in existing row """
    # The document already exist and referenced by id docid
    gpslocation = configfile.get(cam, 'geo')
    camera_id = configfile.get(cam, 'id')
    gpslongitude, gpslatitude = gpslocation.split(",")
    if isinstance(docid, list):
        docidint = docid[0]
    if isinstance(docid, int):
        docidint = docid
        docid = [docid]
    # Check if the document contains lat, long and camid. If yes, then set values,
    # if no, then add values
    document = animaltable.get(doc_id=docidint)
    animaltable.update(set('lat', gpslatitude), doc_ids=docid)
    animaltable.update(set('long', gpslongitude), doc_ids=docid)
    animaltable.update(set('camid', camera_id), doc_ids=docid)
