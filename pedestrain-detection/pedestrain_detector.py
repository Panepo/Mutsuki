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

        assert np.array_equal([1, 1, 200, 7], model.outputs[self.output_blob].shape), (
            "Expected model output shape %s, but got %s"
            % ([1, 1, 200, 7], model.outputs[self.output_blob].shape)
        )

    def preprocess(self, frame):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        assert frame.shape[0] == 1
        assert frame.shape[1] == 3
        input = resize_input(frame, self.input_shape)
        return input

    def enqueue(self, input):
        return super(PedestrainDetector, self).enqueue({self.input_blob: input})

    def start_async(self, frame):
        input = self.preprocess(frame)
        self.enqueue(input)

    def get_detection(self):
        outputs = self.get_outputs()
        results = [PedestrainDetector.Result(out[self.output_blob]) for out in outputs]
        return results
