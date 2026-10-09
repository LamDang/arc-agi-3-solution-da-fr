import torch

from kernel_checks import check


def test_reference_kernel_backward_probe():
    torch.set_num_threads(2)
    result = check("cpu")
    assert max(result["gated_delta_output_gradient_relative_errors"]) == 0
    assert max(result["convolution_output_gradient_relative_errors"]) == 0
