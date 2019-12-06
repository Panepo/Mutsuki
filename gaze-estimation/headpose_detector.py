import numpy as np

from utils import cut_rois, resize_input
from ie_module import Module

class HeadposeDetector(Module):
    POINTS_NUMBER = 35

    class Result:
        def __init__(self, outputs):
            self.points = outputs

            p = lambda i: self[i]
            self.left_eye = p(0)
            self.right_eye = p(3)
            self.nose_tip = p(4)
            self.left_lip_corner = p(8)
            self.right_lip_corner = p(9)

        def __getitem__(self, idx):
            return self.points[idx]

        def get_array(self):
            return np.array(self.points, dtype=np.float64)

    def __init__(self, model):
        super(HeadposeDetector, self).__init__(model)

        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape

        assert np.array_equal(
            [1, self.POINTS_NUMBER * 2], model.outputs[self.output_blob].shape
        ), (
            "Expected model output shape %s, but got %s"
            % ([1, self.POINTS_NUMBER * 2], model.outputs[self.output_blob].shape)
        )

    def preprocess(self, frame, rois):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(HeadposeDetector, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def get_landmarks(self):
        outputs = self.get_outputs()
        results = [
            HeadposeDetector.Result(out[self.output_blob].reshape((-1, 2)))
            for out in outputs
        ]
        return results
