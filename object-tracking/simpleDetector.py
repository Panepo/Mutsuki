import logging as log
import sys
from argparse import ArgumentParser
import cv2
import numpy as np
import time

BREAK_KEYS = {ord("q"), ord("Q"), 27}
CAPTURE_KEYS = {ord("c"), ord("C")}

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
        "-pw",
        "--path_weight",
        metavar="PATH",
        default='./models/yolov3-tiny.weights',
        help="Path to YOLO weights",
    )
    detections.add_argument(
        "-pc",
        "--path_config",
        metavar="PATH",
        default='./models/yolov3-tiny.cfg',
        help="Path to YOLO configs",
    )
    detections.add_argument(
        "-pl",
        "--path_label",
        metavar="PATH",
        default='./models/coco.names',
        help="Path to YOLO labels",
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

    log.info("loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(args.path_config, args.path_weight)
    LABELS = open(args.path_label).read().strip().split("\n")
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

    while(True):
        (grabbed, frame) = cap.read()
        if not grabbed:
            break

        start = time.time()

        if input_crop is not None:
            frame = center_crop(frame, input_crop)

        (H, W) = frame.shape[:2]

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

        cv2.imshow("Detector", frame)

        end = time.time()
        log.info("Detecting took {:.6f} seconds".format(end - start))

        getKey = cv2.waitKey(frame_timeout) & 0xFF
        if getKey in BREAK_KEYS:
            break
        elif getKey in CAPTURE_KEYS:
            log.info("Screen captured")
            save_result(frame, "tracking")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
