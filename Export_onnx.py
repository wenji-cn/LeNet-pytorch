import torch
from LeNet import LeNet
from torch.export import Dim

batch_dim=Dim("batch_size",min=1)

LeNet=LeNet(dim=1,num_classes=10)

state_dict=torch.load("./pth_data/lenet.pth",map_location="cpu")
LeNet.load_state_dict(state_dict)

LeNet.eval()

dummy_input=torch.randn(1,1,28,28)

torch.onnx.export(
    LeNet,
    dummy_input,
    "./onnx_data/lenet.onnx",
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_shapes={
        "x":{0:batch_dim},
    },
)

print("export success!")