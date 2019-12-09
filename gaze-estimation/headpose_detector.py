import numpy as np

from utils import cut_rois, resize_input
from ie_module import Module


class HeadposeDetector(Module):
    class Result:
        def __init__(self, yaw, pitch, roll):
            self.yaw = yaw
            self.pitch = pitch
            self.roll = roll

    def __init__(self, model):
        super(HeadposeDetector, self).__init__(model)

        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 3, "Expected 3 output blob"
        self.input_blob = next(iter(model.inputs))
        self.input_shape = model.inputs[self.input_blob].shape
        self.output_blob_yaw = "angle_y_fc"
        self.output_blob_pitch = "angle_p_fc"
        self.output_blob_roll = "angle_r_fc"

        assert np.array_equal([1, 1], model.outputs[self.output_blob_yaw].shape), (
            "Expected model output shape %s, but got %s"
            % ([1, 1], model.outputs[self.output_blob_yaw].shape)
        )

        assert np.array_equal([1, 1], model.outputs[self.output_blob_pitch].shape), (
            "Expected model output shape %s, but got %s"
            % ([1, 1], model.outputs[self.output_blob_pitch].shape)
        )

        assert np.array_equal([1, 1], model.outputs[self.output_blob_roll].shape), (
            "Expected model output shape %s, but got %s"
            % ([1, 1], model.outputs[self.output_blob_roll].shape)
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

    def get_headposes(self):
        outputs = self.get_outputs()
        results = [
            HeadposeDetector.Result(
                out[self.output_blob_yaw],
                out[self.output_blob_pitch],
                out[self.output_blob_roll],
            )
            for out in outputs
        ]
        return results
