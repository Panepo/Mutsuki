from __future__ import print_function

import logging as log
import os
import sys
import time
from argparse import ArgumentParser, SUPPRESS

import numpy as np
import onnxruntime
from PIL import Image, ImageDraw, ImageFont


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument(
        "-h",
        "--help",
        action="help",
        default=SUPPRESS,
        help="Show this help message and exit.",
    )
    args.add_argument(
        "-m",
        "--model",
        help="Required. Path to an .xml file with a trained model.",
        default="./models/super_resolution.onnx",
        type=str,
        metavar='"<path>"',
    )
    args.add_argument(
        "-i",
        dest="input_source",
        help="Required. Path to an image, video file or a numeric camera ID.",
        default="./images/butterfly_GT_224.png",
        type=str,
        metavar='"<path>"',
    )

    return parser


def main():
    log.basicConfig(
        format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout
    )
    args = build_argparser().parse_args()

    orig_img = Image.open(args.input_source)
    img_ycbcr = orig_img.convert("YCbCr")
    img_y_0, img_cb, img_cr = img_ycbcr.split()
    img_ndarray = np.asarray(img_y_0)

    # Preprocessing Image
    img_4 = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)
    img_5 = img_4.astype(np.float32) / 255.0

    # Run Model on Onnxruntime
    ort_session = onnxruntime.InferenceSession(args.model)
    ort_inputs = {ort_session.get_inputs()[0].name: img_5}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    # Postprocessing Image
    img_out_y = Image.fromarray(
        np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode="L"
    )
    final_img = Image.merge(
        "YCbCr",
        [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ],
    ).convert("RGB")

    fileName = (
        "./output/"
        + "sres"
        + "_"
        + time.strftime("%Y-%m-%d_%H%M%S-", time.localtime())
        + ".png"
    )
    final_img.save(fileName)


if __name__ == "__main__":
    sys.exit(main() or 0)
