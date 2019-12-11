import numpy as np

from utils import cut_rois, resize_input, cut_roi
from ie_module import Module

import cv2
import numpy as np


class GazeDetector(Module):
    class Result:
        def __init__(self, gazeVector):
            self.gazeVector = gazeVector[0]

    class PreProcessResult:
        def __init__(self, imgLeft, imgRight, midLeft, midRight, headposes):
            self.imgLeft = imgLeft
            self.imgRight = imgRight
            self.midLeft = midLeft
            self.midRight = midRight
            self.headposes = []
            self.headposes.append(headposes.yaw)
            self.headposes.append(headposes.pitch)
            self.headposes.append(headposes.roll)

    class BoundingBox:
        def __init__(self, position, size):
            self.position = position
            self.size = size

    class MidPoint:
        def __init__(self, midLeft, midRight):
            self.midLeft = midLeft
            self.midRight = midRight

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

        assert np.array_equal([1, 3], model.outputs[self.output_blob].shape), (
            "Expected model output shape %s, but got %s"
            % ([1, 3], model.outputs[self.output_blob].shape)
        )

    def createEyeBoundingBox(self, eyeLeft, eyeRight, scale=1):
        size = cv2.norm(eyeLeft - eyeRight)
        width = scale * size
        height = width

        midpoint = (eyeLeft + eyeRight) / 2
        x = midpoint[0] - (width / 2)
        y = midpoint[1] - (height / 2)

        position = np.array((x, y))
        size = np.array((width, height))

        return self.BoundingBox(position, size)

    def preprocess(self, frame, rois, landmarks, headposes):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois)

        output = []
        for input, roi, landmark, headpose in zip(inputs, rois, landmarks, headposes):
            boxLeft = self.createEyeBoundingBox(
                landmark.left_eye[0], landmark.left_eye[1]
            )
            boxRight = self.createEyeBoundingBox(
                landmark.right_eye[0], landmark.right_eye[1]
            )

            midLeft = (landmark.left_eye[0] + landmark.left_eye[1]) / 2
            midRight = (landmark.right_eye[0] + landmark.right_eye[1]) / 2

            scaledBoxLeft = self.BoundingBox(
                boxLeft.position * roi.size, boxLeft.size * roi.size
            )
            scaledBoxRight = self.BoundingBox(
                boxRight.position * roi.size, boxRight.size * roi.size
            )

            imgLeft = resize_input(cut_roi(input, scaledBoxLeft), self.input_shape_left)
            imgRight = resize_input(
                cut_roi(input, scaledBoxRight), self.input_shape_right
            )

            output.append(
                self.PreProcessResult(imgLeft, imgRight, midLeft, midRight, headpose)
            )

        return output

    def enqueue(self, input):
        return super(GazeDetector, self).enqueue(
            {
                self.input_blob_left: input.imgLeft,
                self.input_blob_right: input.imgRight,
                self.input_blob_headpose: input.headposes,
            }
        )

    def start_async(self, frame, rois, landmarks, headposes):
        inputs = self.preprocess(frame, rois, landmarks, headposes)
        midpoint = []
        for input in inputs:
            self.enqueue(input)
            midpoint.append(self.MidPoint(input.midLeft, input.midRight))

        return midpoint

    def get_gazevector(self):
        outputs = self.get_outputs()
        results = [GazeDetector.Result(out[self.output_blob]) for out in outputs]
        return results
