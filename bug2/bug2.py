import ai_edge_torch
import tensorflow as tf
import torch
import torch.nn as nn

torch.manual_seed(0)


class Bug2Model(nn.Module):
    def __init__(self):
        super(Bug2Model, self).__init__()
        in_channels = 4
        out_channels = 4
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.layer_norm = nn.LayerNorm(out_channels, eps=1e-6)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # Change to channel last format
        x = self.layer_norm(x)
        return x


def main():
    model = Bug2Model()
    model = model.eval()

    sample_inputs = (torch.randn(1, 4, 4, 4),)
    model = ai_edge_torch.to_channel_last_io(model, args=[0])
    tfl_converter_flags = {"optimizations": [tf.lite.Optimize.DEFAULT]}
    edge_model = ai_edge_torch.convert(
        model,
        sample_args=sample_inputs,
        _ai_edge_converter_flags=tfl_converter_flags,
    )
    edge_model.export("bug2.tflite")

    # Run the model on PyTorch and TFLite
    x = torch.zeros(1, 4, 4, 4)
    print("Execute model on pytorch")
    y = model(x)
    print(y)

    print("Execute model on tflite")
    interpreter = tf.lite.Interpreter(model_path="bug2.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], x.numpy())
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    print(output_data)

    # Check if the outputs are close enough
    if torch.allclose(torch.tensor(output_data), y, atol=1e-5):
        print("Outputs are close enough!")
    else:
        print("Outputs are not close enough!")


if __name__ == "__main__":
    main()
