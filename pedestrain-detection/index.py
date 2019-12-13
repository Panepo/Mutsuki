#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

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

import logging as log
import sys
from argparse import ArgumentParser
from visualizer import Visualizer

DEVICE_KINDS = ["CPU", "GPU", "FPGA", "MYRIAD", "HETERO", "HDDL"]


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group("General")
    general.add_argument(
        "-i",
        "--input",
        metavar="PATH",
        default="2",
        help="(optional) Path to the input video " "('0' for the camera, default)",
    )
    general.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default="",
        help="(optional) Path to save the output video to",
    )
    general.add_argument(
        "--no_show", action="store_true", help="(optional) Do not display output"
    )
    general.add_argument(
        "-tl",
        "--timelapse",
        action="store_true",
        help="(optional) Auto-pause after each frame",
    )
    general.add_argument(
        "-cw",
        "--crop_width",
        default=0,
        type=int,
        help="(optional) Crop the input stream to this width "
        "(default: no crop). Both -cw and -ch parameters "
        "should be specified to use crop.",
    )
    general.add_argument(
        "-ch",
        "--crop_height",
        default=0,
        type=int,
        help="(optional) Crop the input stream to this height "
        "(default: no crop). Both -cw and -ch parameters "
        "should be specified to use crop.",
    )

    models = parser.add_argument_group("Models")
    models.add_argument(
        "-m_pd",
        metavar="PATH",
        default="./models/pedestrian-and-vehicle-detector-adas-0001.xml",
        help="Path to the Pedestrain Detection model XML file",
    )

    infer = parser.add_argument_group("Inference options")
    infer.add_argument(
        "-d_pd",
        default="CPU",
        choices=DEVICE_KINDS,
        help="(optional) Target device for the "
        "Pedestrain Detection model (default: %(default)s)",
    )
    infer.add_argument(
        "-l",
        "--cpu_lib",
        metavar="PATH",
        default="/opt/intel/openvino_2019.3.376/inference_engine/lib/intel64/libcpu_extension_sse4.so",
        help="(optional) For MKLDNN (CPU)-targeted custom layers, if any. "
        "Path to a shared library with custom layers implementations",
    )
    infer.add_argument(
        "-c",
        "--gpu_lib",
        metavar="PATH",
        default="",
        help="(optional) For clDNN (GPU)-targeted custom layers, if any. "
        "Path to the XML file with descriptions of the kernels",
    )
    infer.add_argument(
        "-v", "--verbose", action="store_true", help="(optional) Be more verbose"
    )
    infer.add_argument(
        "-pc",
        "--perf_stats",
        action="store_true",
        help="(optional) Output detailed per-layer performance stats",
    )

    return parser


def main():
    args = build_argparser().parse_args()

    log.basicConfig(
        format="[ %(levelname)s ] %(asctime)-15s %(message)s",
        level=log.INFO if not args.verbose else log.DEBUG,
        stream=sys.stdout,
    )

    log.debug(str(args))

    visualizer = Visualizer(args)
    visualizer.run(args)


if __name__ == "__main__":
    main()
