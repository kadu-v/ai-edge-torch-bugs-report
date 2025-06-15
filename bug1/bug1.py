import ai_edge_torch
import tensorflow as tf
import torch
import torch.nn as nn


class Bug1Model(nn.Module):
    def __init__(self):
        super(Bug1Model, self).__init__()
        self.fc = nn.Linear(10, 5, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        return x


def main():
    model = Bug1Model()
    model = model.eval()

    sample_inputs = (torch.randn(1, 64, 10),)
    tfl_converter_flags = {"optimizations": [tf.lite.Optimize.DEFAULT]}
    edge_model = ai_edge_torch.convert(
        model, sample_args=sample_inputs, _ai_edge_converter_flags=tfl_converter_flags
    )
    edge_model.export("bug1.tflite")


if __name__ == "__main__":
    main()
