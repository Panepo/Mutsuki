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
import os.path as osp
import sys
import time
import math

import cv2
import numpy as np

from openvino.inference_engine import IENetwork
from ie_module import InferenceContext
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from headpose_detector import HeadposeDetector
from gaze_detector import GazeDetector

DEVICE_KINDS = ["CPU", "GPU", "FPGA", "MYRIAD", "HETERO", "HDDL"]


class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        used_devices = set([args.d_fd, args.d_lm])
        self.context = InferenceContext()
        context = self.context
        context.load_plugins(used_devices, args.cpu_lib, args.gpu_lib)
        for d in used_devices:
            context.get_plugin(d).set_config(
                {"PERF_COUNT": "YES" if args.perf_stats else "NO"}
            )

        log.info("Loading models")
        face_detector_net = self.load_model(args.m_fd)
        landmarks_net = self.load_model(args.m_lm)
        headpose_net = self.load_model(args.m_hp)
        gaze_net = self.load_model(args.m_gz)

        self.face_detector = FaceDetector(
            face_detector_net,
            confidence_threshold=args.t_fd,
            roi_scale_factor=args.exp_r_fd,
        )
        self.landmarks_detector = LandmarksDetector(landmarks_net)
        self.headpose_detector = HeadposeDetector(headpose_net)
        self.gaze_detector = GazeDetector(gaze_net)

        self.face_detector.deploy(args.d_fd, context)
        self.landmarks_detector.deploy(args.d_lm, context, queue_size=self.QUEUE_SIZE)
        self.headpose_detector.deploy(args.d_hp, context)
        self.gaze_detector.deploy(args.d_gz, context)
        log.info("Models are loaded")

    def load_model(self, model_path):
        model_path = osp.abspath(model_path)
        model_description_path = model_path
        model_weights_path = osp.splitext(model_path)[0] + ".bin"
        log.info("Loading the model from '%s'" % (model_description_path))
        assert osp.isfile(
            model_description_path
        ), "Model description is not found at '%s'" % (model_description_path)
        assert osp.isfile(model_weights_path), "Model weights are not found at '%s'" % (
            model_weights_path
        )
        model = IENetwork(model_description_path, model_weights_path)
        log.info("Model is loaded")
        return model

    def process(self, frame):
        assert len(frame.shape) == 3, "Expected input frame in (H, W, C) format"
        assert frame.shape[2] in [3, 4], "Expected BGR or BGRA input"

        frame = frame.transpose((2, 0, 1))  # HWC to CHW
        frame = np.expand_dims(frame, axis=0)

        self.face_detector.clear()
        self.landmarks_detector.clear()
        self.headpose_detector.clear()

        self.face_detector.start_async(frame)
        rois = self.face_detector.get_roi_proposals(frame)
        if self.QUEUE_SIZE < len(rois):
            log.warning(
                "Too many faces for processing."
                " Will be processed only %s of %s." % (self.QUEUE_SIZE, len(rois))
            )
            rois = rois[: self.QUEUE_SIZE]
        self.landmarks_detector.start_async(frame, rois)
        landmarks = self.landmarks_detector.get_landmarks()
        self.headpose_detector.start_async(frame, rois)
        headposes = self.headpose_detector.get_headposes()
        midpoints = self.gaze_detector.start_async(frame, rois, landmarks, headposes)
        gazevectors = self.gaze_detector.get_gazevector()

        outputs = [rois, landmarks, headposes, gazevectors, midpoints]

        return outputs

    def get_performance_stats(self):
        stats = {
            "face_detector": self.face_detector.get_performance_stats(),
            "landmarks": self.landmarks_detector.get_performance_stats(),
            "headpose_detector": self.headpose_detector.get_performance_stats(),
            "gaze_detector": self.gaze_detector.get_performance_stats(),
        }
        return stats


class Visualizer:
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord("q"), ord("Q"), 27}
    CAPTURE_KEYS = {ord("c"), ord("C")}
    DETECT_KEYS = {ord("d"), ord("D")}
    POSE_KEYS = {ord("h"), ord("H")}
    LANDMARK_KEYS = {ord("l"), ord("L")}
    GAZE_KEYS = {ord("g"), ord("G")}
    TOGGLE_KEYS = {ord("a"), ord("A")}
    TOGGLE_OFF_KEYS = {ord("n"), ord("N")}

    def __init__(self, args):
        self.frame_processor = FrameProcessor(args)
        self.display = not args.no_show
        self.print_perf_stats = args.perf_stats

        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.frame_num = 0
        self.frame_count = -1

        self.input_crop = None
        if args.crop_width and args.crop_height:
            self.input_crop = np.array((args.crop_width, args.crop_height))

        self.frame_timeout = 0 if args.timelapse else 1

        self.toggle_detect = True
        self.toggle_pose = True
        self.toggle_landmark = True
        self.toggle_gaze = True

    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_text_with_background(
        self,
        frame,
        text,
        origin,
        font=cv2.FONT_HERSHEY_DUPLEX,
        scale=2,
        color=(255, 255, 255),
        thickness=1,
        bgcolor=(255, 0, 0),
    ):
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(
            frame,
            tuple((origin + (0, baseline)).astype(int)),
            tuple((origin + (text_size[0], -text_size[1])).astype(int)),
            bgcolor,
            cv2.FILLED,
        )
        cv2.putText(
            frame, text, tuple(origin.astype(int)), font, scale, color, thickness
        )
        return text_size, baseline

    # Draw face ROI border
    def draw_detection_roi(self, frame, roi):

        cv2.rectangle(
            frame, tuple(roi.position), tuple(roi.position + roi.size), (0, 220, 0), 2
        )

    # Draw face landmarks
    def draw_detection_keypoints(self, frame, roi, landmarks):
        keypoints = landmarks.points

        for point in keypoints:
            center = roi.position + roi.size * point
            cv2.circle(frame, tuple(center.astype(int)), 2, (0, 255, 255), 2)

    # Draw visualation of headposes
    def draw_detection_headposes(self, frame, roi, headposes):
        sinY = math.sin(headposes.yaw * math.pi / 180)
        sinP = math.sin(headposes.pitch * math.pi / 180)
        sinR = math.sin(headposes.roll * math.pi / 180)

        cosY = math.cos(headposes.yaw * math.pi / 180)
        cosP = math.cos(headposes.pitch * math.pi / 180)
        cosR = math.cos(headposes.roll * math.pi / 180)

        axisLength = 0.4 * roi.size[0]
        xCenter = roi.position[0] + roi.size[0] / 2
        yCenter = roi.position[1] + roi.size[1] / 2

        cv2.line(
            frame,
            (int(xCenter), int(yCenter)),
            (
                int(xCenter + axisLength * (cosR * cosY + sinY * sinP * sinR)),
                int(yCenter + axisLength * (cosP * sinR)),
            ),
            (0, 0, 255),
            2,
        )
        cv2.line(
            frame,
            (int(xCenter), int(yCenter)),
            (
                int(xCenter + axisLength * (cosR * sinY * sinP + cosY * sinR)),
                int(yCenter - axisLength * (cosP * cosR)),
            ),
            (0, 255, 0),
            2,
        )
        cv2.line(
            frame,
            (int(xCenter), int(yCenter)),
            (
                int(xCenter + axisLength * (sinY * cosP)),
                int(yCenter + axisLength * sinP),
            ),
            (255, 0, 255),
            2,
        )

    # Draw visualation of gaze vectors
    def draw_detection_gaze(self, frame, roi, gazevector, midpoints):
        eyeLeft = roi.position + roi.size * midpoints.midLeft
        eyeRight = roi.position + roi.size * midpoints.midRight

        arrowLength = 0.4 * roi.size[0]
        gazeArrowLeftX = int(eyeLeft[0] + gazevector.gazeVector[0] * arrowLength)
        gazeArrowLeftY = int(eyeLeft[1] - gazevector.gazeVector[1] * arrowLength)

        gazeArrowRightX = int(eyeRight[0] + gazevector.gazeVector[0] * arrowLength)
        gazeArrowRightY = int(eyeRight[1] - gazevector.gazeVector[1] * arrowLength)

        cv2.arrowedLine(
            frame,
            tuple(eyeLeft.astype(int)),
            (gazeArrowLeftX, gazeArrowLeftY),
            (255, 0, 0),
            2,
        )
        cv2.arrowedLine(
            frame,
            tuple(eyeRight.astype(int)),
            (gazeArrowRightX, gazeArrowRightY),
            (255, 0, 0),
            2,
        )

    def draw_detections(self, frame, detections):
        for roi, landmarks, headposes, gazevectors, midpoints in zip(*detections):
            if self.toggle_detect:
                self.draw_detection_roi(frame, roi)

            if self.toggle_landmark:
                self.draw_detection_keypoints(frame, roi, landmarks)

            if self.toggle_pose:
                self.draw_detection_headposes(frame, roi, headposes)

            if self.toggle_gaze:
                self.draw_detection_gaze(frame, roi, gazevectors, midpoints)

    def draw_status(self, frame, detections):
        origin = np.array([10, 10])
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text_size, _ = self.draw_text_with_background(
            frame,
            "Frame time: %.3fs" % (self.frame_time),
            origin,
            font,
            text_scale,
            color,
        )
        self.draw_text_with_background(
            frame,
            "FPS: %.1f" % (self.fps),
            (origin + (0, text_size[1] * 1.5)),
            font,
            text_scale,
            color,
        )

        log.debug(
            "Frame: %s/%s, detections: %s, "
            "frame time: %.3fs, fps: %.1f"
            % (
                self.frame_num,
                self.frame_count,
                len(detections[-1]),
                self.frame_time,
                self.fps,
            )
        )

        if self.print_perf_stats:
            log.info("Performance stats:")
            log.info(self.frame_processor.get_performance_stats())

    def display_interactive_window(self, frame):
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.5
        text = "Press '%s' key to exit" % (self.BREAK_KEY_LABELS)
        thickness = 2
        text_size = cv2.getTextSize(text, font, text_scale, thickness)
        origin = np.array([frame.shape[-2] - text_size[0][0] - 10, 10])
        line_height = np.array([0, text_size[0][1]]) * 1.5
        cv2.putText(
            frame, text, tuple(origin.astype(int)), font, text_scale, color, thickness
        )

        cv2.imshow("Face recognition demo", frame)

    def save_result(self, image, name):
        fileName = (
            "./output/"
            + name
            + "_"
            + time.strftime("%Y-%m-%d_%H%M%S-", time.localtime())
            + ".png"
        )
        cv2.imwrite(fileName, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        log.info("saved results to {}".format(fileName))

    def keyboard_command(self, frame):
        key = cv2.waitKey(self.frame_timeout) & 0xFF

        if key in self.CAPTURE_KEYS:
            log.info("Screen captured")
            self.save_result(frame, "recognition")
        elif key in self.DETECT_KEYS:
            self.toggle_detect = not self.toggle_detect
        elif key in self.POSE_KEYS:
            self.toggle_pose = not self.toggle_pose
        elif key in self.LANDMARK_KEYS:
            self.toggle_landmark = not self.toggle_landmark
        elif key in self.GAZE_KEYS:
            self.toggle_gaze = not self.toggle_gaze
        elif key in self.TOGGLE_KEYS:
            self.toggle_detect = True
            self.toggle_gaze = True
            self.toggle_landmark = True
            self.toggle_pose = True
        elif key in self.TOGGLE_OFF_KEYS:
            self.toggle_detect = False
            self.toggle_gaze = False
            self.toggle_landmark = False
            self.toggle_pose = False

        return key in self.BREAK_KEYS

    def process(self, input_stream, output_stream):
        self.input_stream = input_stream
        self.output_stream = output_stream

        while input_stream.isOpened():
            has_frame, frame = input_stream.read()
            if not has_frame:
                break

            if self.input_crop is not None:
                frame = Visualizer.center_crop(frame, self.input_crop)
            detections = self.frame_processor.process(frame)

            self.draw_detections(frame, detections)
            self.draw_status(frame, detections)

            if output_stream:
                output_stream.write(frame)
            if self.display:
                self.display_interactive_window(frame)
                if self.keyboard_command(frame):
                    break

            self.update_fps()
            self.frame_num += 1

    @staticmethod
    def center_crop(frame, crop_size):
        fh, fw, fc = frame.shape
        crop_size[0] = min(fw, crop_size[0])
        crop_size[1] = min(fh, crop_size[1])
        return frame[
            (fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
            (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
            :,
        ]

    def run(self, args):
        input_stream = Visualizer.open_input_stream(args.input)
        if input_stream is None or not input_stream.isOpened():
            log.error("Cannot open input stream: %s" % args.input)
        fps = input_stream.get(cv2.CAP_PROP_FPS)
        frame_size = (
            int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.crop_width and args.crop_height:
            crop_size = (args.crop_width, args.crop_height)
            frame_size = tuple(np.minimum(frame_size, crop_size))
        log.info(
            "Input stream info: %d x %d @ %.2f FPS"
            % (frame_size[0], frame_size[1], fps)
        )
        output_stream = Visualizer.open_output_stream(args.output, fps, frame_size)

        self.process(input_stream, output_stream)

        # Release resources
        if output_stream:
            output_stream.release()
        if input_stream:
            input_stream.release()

        cv2.destroyAllWindows()

    @staticmethod
    def open_input_stream(path):
        log.info("Reading input data from '%s'" % (path))
        stream = path
        try:
            stream = int(path)
        except ValueError:
            pass
        return cv2.VideoCapture(stream)

    @staticmethod
    def open_output_stream(path, fps, frame_size):
        output_stream = None
        if path != "":
            if not path.endswith(".avi"):
                log.warning(
                    "Output file extension is not 'avi'. "
                    "Some issues with output can occur, check logs."
                )
            log.info("Writing output to '%s'" % (path))
            output_stream = cv2.VideoWriter(
                path, cv2.VideoWriter.fourcc(*"MJPG"), fps, frame_size
            )
        return output_stream
