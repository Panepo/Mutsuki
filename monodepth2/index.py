# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import logging as log
import time

import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple testing funtion for Monodepthv2 models."
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="path to a test image or folder of images",
        default=2,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="name of a pretrained model to use",
        default="mono_640x192",
        choices=[
            "mono_640x192",
            "stereo_640x192",
            "mono+stereo_640x192",
            "mono_no_pt_640x192",
            "stereo_no_pt_640x192",
            "mono+stereo_no_pt_640x192",
            "mono_1024x320",
            "stereo_1024x320",
            "mono+stereo_1024x320",
        ],
    )
    parser.add_argument(
        "--ext", type=str, help="image extension to search for in folder", default="jpg"
    )

    return parser.parse_args()


# Function to predict for a single image or folder of images
def main(args):
    log.basicConfig(
        format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout
    )

    device = torch.device("cpu")

    model_path = os.path.join("models", args.model_name)
    log.info("-> Loading model from " + model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    log.info("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc["height"]
    feed_width = loaded_dict_enc["width"]
    filtered_dict_enc = {
        k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()
    }
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    log.info("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4)
    )

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    try:
        input_source = int(args.input)
    except ValueError:
        input_source = args.input
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        log.error('Failed to open "{}"'.format(args.input))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    log.info("Starting inference...")
    print(
        "To close the application, press 'CTRL+C' here or switch to the output window and press ESC key"
    )
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Load frame and preprocess
        input_image = pil.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        inf_start = time.time()
        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp,
            (original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )
        inf_end = time.time()
        det_time = inf_end - inf_start

        # colorize depth image
        render_start = time.time()
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(
            np.uint8
        )
        output = pil.fromarray(colormapped_im)

        # Show resulting image.
        result = cv2.cvtColor(np.asarray(output),cv2.COLOR_RGB2BGR)
        cv2.imshow("Results", result)
        render_end = time.time()
        render_time = render_end - render_start

        key = cv2.waitKey(1) & 0xFF
        if key in {ord("q"), ord("Q"), 27}:
            break
        elif key in {ord("c"), ord("C")}:
            fileName = (
                "./output/"
                + "segmentation"
                + "_"
                + time.strftime("%Y-%m-%d_%H%M%S-", time.localtime())
                + ".png"
            )
            cv2.imwrite(fileName, result, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            log.info("saved results to {}".format(fileName))


if __name__ == "__main__":
    args = parse_args()
    main(args)
