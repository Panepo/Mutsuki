import logging as log
import sys
from argparse import ArgumentParser
import cv2
import numpy as np
import time

TRACKER_KINDS = ["KCF", "BOOSTING", "MIL", "TLD", "MEDIANFLOW", "GOTURN"]

BREAK_KEYS = {ord("q"), ord("Q"), 27}
CAPTURE_KEYS = {ord("c"), ord("C")}
TRACKING_KEYS = {ord("t"), ord("T")}


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group("General")
    general.add_argument(
        "-i",
        "--input",
        metavar="PATH",
        default="0",
        help="(optional) Path to the input video " "('0' for the camera, default)",
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

    detections = parser.add_argument_group("Detections")
    detections.add_argument(
        "-t",
        "--tracker",
        type=str,
        choices=TRACKER_KINDS,
        default="KCF",
        help="OpenCV object tracker type",
    )

    return parser


def center_crop(frame, crop_size):
    fh, fw, fc = frame.shape
    crop_size[0] = min(fw, crop_size[0])
    crop_size[1] = min(fh, crop_size[1])
    return frame[
        (fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
        (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
        :,
    ]


def save_result(image, name):
    fileName = (
        "./output/"
        + name
        + "_"
        + time.strftime("%Y-%m-%d_%H%M%S-", time.localtime())
        + ".png"
    )
    cv2.imwrite(fileName, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    log.info("saved results to {}".format(fileName))


def main():
    args = build_argparser().parse_args()

    log.basicConfig(
        format="[ %(levelname)s ] %(asctime)-15s %(message)s",
        level=log.INFO,
        stream=sys.stdout,
    )
    log.debug(str(args))

    cap = cv2.VideoCapture(args.input)
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    input_crop = None
    if args.crop_width and args.crop_height:
        input_crop = np.array((args.crop_width, args.crop_height))
        crop_size = (args.crop_width, args.crop_height)
        frame_size = tuple(np.minimum(frame_size, crop_size))

    frame_timeout = 0 if args.timelapse else 1

    if args.tracker == "BOOSTING":
        tracker = cv2.TrackerBoosting_create()
    elif args.tracker == "MIL":
        tracker = cv2.TrackerMIL_create()
    elif args.tracker == "KCF":
        tracker = cv2.TrackerKCF_create()
    elif args.tracker == "TLD":
        tracker = cv2.TrackerTLD_create()
    elif args.tracker == "MEDIANFLOW":
        tracker = cv2.TrackerMedianFlow_create()
    elif args.tracker == "GOTURN":
        tracker = cv2.TrackerGOTURN_create()

    traceStart = 0

    while True:
        (grabbed, frame) = cap.read()
        if not grabbed:
            log.error("no inputs")
            break

        start = time.time()

        if input_crop is not None:
            frame = center_crop(frame, input_crop)

        if traceStart:
            ok, bbox = tracker.update(frame)

            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)

            end = time.time()
            log.info("Tracking took {:.6f} seconds".format(end - start))

        cv2.imshow("Tracking", frame)

        getKey = cv2.waitKey(frame_timeout) & 0xFF
        if getKey in BREAK_KEYS:
            break
        elif getKey in TRACKING_KEYS:
            bbox = cv2.selectROI(frame, False)
            print(bbox)
            ok = tracker.init(frame, bbox)
            traceStart = 1
            cv2.destroyAllWindows()
        elif getKey in CAPTURE_KEYS:
            log.info("Screen captured")
            save_result(frame, "tracking")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
