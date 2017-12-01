# classify_image
Animal image classification with Inception

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top prediction along with their probabilities.

use the --help parameter for all required and optional arguments:
python classify_image.py --help

Standard call:
python classify_image.py --with_cam cam0x --image_file file.jpeg
