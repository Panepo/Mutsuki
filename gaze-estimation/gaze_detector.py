import numpy as np

from utils import cut_rois, resize_input
from ie_module import Module

import cv2
import numpy as np

class GazeDetector(Module):
    class Result:
        def __init__(self, yaw, pitch, roll):
            self.yaw = yaw
            self.pitch = pitch
            self.roll = roll

    class BoundingBox:
        def __init__(self, x, y, width, height):
            self.x = x
            self.y = y
            self.width = width
            self.height = height

    def __init__(self, model):
        super(GazeDetector, self).__init__(model)

        assert len(model.inputs) == 3, "Expected 3 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        self.input_blob_left = "left_eye_image"
        self.input_blob_right = "right_eye_image"
        self.input_blob_headpose = "head_pose_angles"

        self.input_shape_left = model.inputs[self.input_blob_left].shape
        self.input_shape_right = model.inputs[self.input_blob_right].shape

        self.output_blob = next(iter(model.outputs))
        self.output_shape = model.outputs[self.output_blob].shape

        assert np.array_equal(
            [1, 3], model.outputs[self.output_blob].shape
        ), (
            "Expected model output shape %s, but got %s"
            % ([1, 3], model.outputs[self.output_blob].shape)
        )

    def createEyeBoundingBox(self, eyeLeft, eyeRight, scale = 1):
        print(eyeLeft)


    def preprocess(self, frame, rois, landmarks, headposes):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois)

        output = []
        for input, landmark in zip(inputs, landmarks):
            self.createEyeBoundingBox(landmark.left_eye[0], landmark.left_eye[1])
            output.append(resize_input(input, self.input_shape))

        return output

    def enqueue(self, input):
        return super(GazeDetector, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois, landmarks, headposes):
        inputs = self.preprocess(frame, rois, landmarks, headposes)
        for input in inputs:
            self.enqueue(input)

    def get_headposes(self):
        outputs = self.get_outputs()
        results = [
            GazeDetector.Result(
                out[self.output_blob_yaw],
                out[self.output_blob_pitch],
                out[self.output_blob_roll],
            )
            for out in outputs
        ]
        return results
