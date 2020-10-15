import numpy as np
import torch
import tqdm
from torch import nn


class MyLayerNormNoAffine(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x):
        assert len(x.shape) == 4
        d = len(self.normalized_shape)
        assert x.shape[-d:] == self.normalized_shape

        # Last d dims. In the special case of d=3 and spatial input, this corresponds to
        # all dims except B.
        mu = x.mean(dim=tuple(range(4 - d, 4)), keepdim=True)
        sigma = x.var(dim=tuple(range(4 - d, 4)), keepdim=True, unbiased=False)

        return (x - mu) / torch.sqrt(sigma + self.eps)


def test_LayerNorm():
    np.random.seed(42)
    torch.manual_seed(42)

    input_shape = (8, 4, 20, 16)
    for _ in tqdm.trange(20, desc=f'LayerNorm()'):
        # layers initialization
        n_in = input_shape[np.random.choice([1, 2, 3]):]
        torch_layer = nn.LayerNorm(n_in, elementwise_affine=False)
        custom_layer = MyLayerNormNoAffine(n_in)

        for _ in range(10):
            torch_layer.train()
            custom_layer.train()

            layer_input = np.random.uniform(-5, 5, input_shape).astype(np.float32)
            torch_layer_input = torch.tensor(layer_input, requires_grad=True)
            custom_layer_input = torch.tensor(layer_input, requires_grad=True)
            next_layer_grad = torch.from_numpy(np.random.uniform(-5, 5, input_shape).astype(np.float32))

            # 1. check layer output
            torch_layer_output = torch_layer(torch_layer_input)
            custom_layer_output = custom_layer(custom_layer_input)
            assert torch.allclose(torch_layer_output, custom_layer_output, atol=1e-6)

            # 2. check layer input grad
            torch_layer_output.backward(next_layer_grad)
            custom_layer_output.backward(next_layer_grad)
            assert torch.allclose(torch_layer_input.grad, custom_layer_input.grad, atol=1e-5)

            # 4. check evaluation mode
            torch_layer.eval()
            custom_layer.eval()
            torch_layer_output = torch_layer(torch_layer_input)
            custom_layer_output = custom_layer(custom_layer_input)
            assert torch.allclose(torch_layer_output, custom_layer_output, atol=1e-6)
