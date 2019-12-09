import numpy as np

from utils import cut_rois, resize_input
from ie_module import Module


class GazeDetector(Module):
    class Result:
        def __init__(self, yaw, pitch, roll):
            self.yaw = yaw
            self.pitch = pitch
            self.roll = roll

    def __init__(self, model):
        super(GazeDetector, self).__init__(model)

        assert len(model.inputs) == 2, "Expected 2 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        self.input_blob_left = "left_eye_image"
        self.input_blob_right = "right_eye_image"

        self.output_blob = next(iter(model.outputs))
        self.output_shape = model.outputs[self.output_blob].shape

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
        return super(GazeDetector, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
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
