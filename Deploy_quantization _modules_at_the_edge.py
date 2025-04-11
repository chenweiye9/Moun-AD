from torch.quantization import QuantStub, DeQuantStub


class EdgeQuantizedUNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.model = base_model

        self.quant_config = {
            'attention': torch.bfloat16,
            'conv': torch.int8
        }

    def quantize(self, tensor, layer_type):
        if self.quant_config[layer_type] == torch.int8:
            return torch.quantize_per_tensor(tensor, 0.1, 128, torch.qint8)
        else:
            return tensor.to(self.quant_config[layer_type])

    def forward(self, x):
        x = self.quant(x)

        for name, module in self.model.named_children():
            if 'attention' in name:
                x = self.quantize(x, 'attention')
            elif 'conv' in name:
                x = self.quantize(x, 'conv')
            x = module(x)

        return self.dequant(x)


quant_model = EdgeQuantizedUNet(base_model)
quant_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(quant_model, inplace=True)

trt_model = torch2trt(
    quant_model,
    [example_input],
    fp16_mode=True,
    max_workspace_size=1 << 30
)