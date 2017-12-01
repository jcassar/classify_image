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

import sys
import ClassifyImage.initialize as initialize

if __name__ == '__main__':
    initialize.classify_run(sys.argv)
