import numpy as np

from utils import cut_rois, resize_input
from ie_module import Module

import cv2
import numpy as np


class PedestrainDetector(Module):
    class Result:
        def __init__(self, gazeVector):
            self.gazeVector = gazeVector[0]

    def __init__(self, model):
        super(PedestrainDetector, self).__init__(model)

        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape
        self.output_shape = model.outputs[self.output_blob].shape

        assert (
            len(self.output_shape) == 4
            and self.output_shape[3] == self.Result.OUTPUT_SIZE
        ), "Expected model output shape with %s outputs" % (self.Result.OUTPUT_SIZE)

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
        assert frame.shape[0] == 1
        assert frame.shape[1] == 3
        input = resize_input(frame, self.input_shape)
        return input

    def enqueue(self, input):
        return super(PedestrainDetector, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois, landmarks, headposes):
        input = self.preprocess(frame)
        self.enqueue(input)

    '''
    def get_gazevector(self):
        outputs = self.get_outputs()
        results = [GazeDetector.Result(out[self.output_blob]) for out in outputs]
        return results
    '''
