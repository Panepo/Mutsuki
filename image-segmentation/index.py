#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
import time
from openvino.inference_engine import IENetwork, IECore

classes_color_map = [
    (150, 150, 150),
    (58, 55, 169),
    (211, 51, 17),
    (157, 80, 44),
    (23, 95, 189),
    (210, 133, 34),
    (76, 226, 202),
    (101, 138, 127),
    (223, 91, 182),
    (80, 128, 113),
    (235, 155, 55),
    (44, 151, 243),
    (159, 80, 170),
    (239, 208, 44),
    (128, 50, 51),
    (82, 141, 193),
    (9, 107, 10),
    (223, 90, 142),
    (50, 248, 83),
    (178, 101, 130),
    (71, 30, 204),
]


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
        help="Required. Path to an .xml file with a trained model",
        default="./models/road-segmentation-adas-0001.xml",
        type=str,
    )
    args.add_argument(
        "-i",
        "--input",
        metavar="PATH",
        default="2",
        help="(optional) Path to the input video " "('0' for the camera, default)",
    )
    args.add_argument(
        "-l",
        "--cpu_extension",
        help="Optional. Required for CPU custom layers. "
        "Absolute MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the "
        "kernels implementations",
        type=str,
        default="/opt/intel/openvino_2019.3.376/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    )
    args.add_argument(
        "-d",
        "--device",
        help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
        "acceptable. Sample will look for a suitable plugin for device specified. Default value is CPU",
        default="CPU",
        type=str,
    )
    args.add_argument(
        "-nt",
        "--number_top",
        help="Optional. Number of top results",
        default=10,
        type=int,
    )
    args.add_argument(
        "--no_keep_aspect_ratio",
        help="Optional. Force image resize not to keep aspect ratio.",
        action="store_true",
    )
    return parser


def main():
    log.basicConfig(
        format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout
    )
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    log.info("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension and "CPU" in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [
            l for l in net.layers.keys() if l not in supported_layers
        ]
        if len(not_supported_layers) != 0:
            log.error(
                "Following layers are not supported by the plugin for specified device {}:\n {}".format(
                    args.device, ", ".join(not_supported_layers)
                )
            )
            log.error(
                "Please try to specify cpu extensions library path in sample's command line parameters using -l "
                "or --cpu_extension command line argument"
            )
            sys.exit(1)
    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = len(args.input)
    n, c, h, w = net.inputs[input_blob].shape
    images = np.ndarray(shape=(n, c, h, w))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)
    del net

    try:
        input_source = int(args.input)
    except ValueError:
        input_source = args.input
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        log.error('Failed to open "{}"'.format(args.input))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    render_time = 0


    log.info("Starting inference...")
    print(
        "To close the application, press 'CTRL+C' here or switch to the output window and press ESC key"
    )
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame.shape[:-1] != (h, w):
            image = cv2.resize(frame, (w, h))

        # Change data layout from HWC to CHW.
        imageT = image.transpose((2, 0, 1))
        images[0] = imageT

        # Run the net.
        inf_start = time.time()
        res = exec_net.infer(inputs={input_blob: images})
        inf_end = time.time()
        det_time = inf_end - inf_start

        render_start = time.time()

        res = res[out_blob]
        if len(res.shape) == 3:
            res = np.expand_dims(res, axis=1)
        if len(res.shape) == 4:
            _, _, out_h, out_w = res.shape
        else:
            raise Exception(
                "Unexpected output blob shape {}. Only 4D and 3D output blobs are supported".format(
                    res.shape
                )
            )

        classes_map = np.zeros(shape=(out_h, out_w, 3), dtype=np.uint8)
        for i in range(out_h):
            for j in range(out_w):
                if len(res[0][:, i, j]) == 1:
                    pixel_class = int(res[0][:, i, j])
                else:
                    pixel_class = np.argmax(res[0][:, i, j])
                classes_map[i, j, :] = classes_color_map[min(pixel_class, 20)]

        #render_start = time.time()

        # Combine mask into image
        cv2.addWeighted(image, 0.7, classes_map, 0.3, 0, dst=image)

        # Draw performance stats.
        inf_time_message = "Inference time: {:.3f} ms".format(det_time * 1000)
        render_time_message = "OpenCV rendering time: {:.3f} ms".format(
            render_time * 1000
        )
        cv2.putText(
            image,
            inf_time_message,
            (15, 15),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (200, 10, 10),
            1,
        )
        cv2.putText(
            image,
            render_time_message,
            (15, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (10, 10, 200),
            1,
        )


        # Show resulting image.
        cv2.imshow("Results", image)
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
            cv2.imwrite(fileName, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            log.info("saved results to {}".format(fileName))

    cv2.destroyAllWindows()
    cap.release()
    del exec_net
    del ie

if __name__ == "__main__":
    sys.exit(main() or 0)
