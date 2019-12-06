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

import cv2
import numpy as np

from openvino.inference_engine import IENetwork
from ie_module import InferenceContext
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector

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

        self.face_detector = FaceDetector(
            face_detector_net,
            confidence_threshold=args.t_fd,
            roi_scale_factor=args.exp_r_fd,
        )
        self.landmarks_detector = LandmarksDetector(landmarks_net)

        self.face_detector.deploy(args.d_fd, context)
        self.landmarks_detector.deploy(args.d_lm, context, queue_size=self.QUEUE_SIZE)
        log.info("Models are loaded")

        self.allow_grow = args.allow_grow and not args.no_show

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

        orig_image = frame.copy()
        frame = frame.transpose((2, 0, 1))  # HWC to CHW
        frame = np.expand_dims(frame, axis=0)

        self.face_detector.clear()
        self.landmarks_detector.clear()

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

        outputs = [rois, landmarks]

        return outputs

    def get_performance_stats(self):
        stats = {
            "face_detector": self.face_detector.get_performance_stats(),
            "landmarks": self.landmarks_detector.get_performance_stats()
        }
        return stats


class Visualizer:
    BREAK_KEY_LABELS = "q(Q) or Escape"
    BREAK_KEYS = {ord("q"), ord("Q"), 27}
    CAPTURE_KEYS = {ord("c"), ord("C")}

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

    def draw_detection_roi(self, frame, roi):
        # Draw face ROI border
        cv2.rectangle(
            frame, tuple(roi.position), tuple(roi.position + roi.size), (0, 220, 0), 2
        )

    def draw_detection_keypoints(self, frame, roi, landmarks):
        keypoints = [
            landmarks.left_eye,
            landmarks.right_eye,
            landmarks.nose_tip,
            landmarks.left_lip_corner,
            landmarks.right_lip_corner,
        ]

        for point in keypoints:
            center = roi.position + roi.size * point
            cv2.circle(frame, tuple(center.astype(int)), 2, (0, 255, 255), 2)

    def draw_detections(self, frame, detections):
        for roi, landmarks in zip(*detections):
            self.draw_detection_roi(frame, roi)
            self.draw_detection_keypoints(frame, roi, landmarks)

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
            "../output/"
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
