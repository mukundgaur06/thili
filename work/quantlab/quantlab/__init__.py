"""QuantLab: train a small CNN on Fashion-MNIST, quantize it five ways, and measure the trade-offs."""

import warnings

# torch.ao.quantization (eager mode) still works but warns that it will move to torchao.
warnings.filterwarnings("ignore", message=".*torch.ao.quantization is deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.ao")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=r"(?s).*torchao.*")                 # the eager API's migration notice
warnings.filterwarnings("ignore", module=r"torch\.onnx.*")
warnings.filterwarnings("ignore", message=r"(?s).*_get_name.*|.*__torch_function__.*")
warnings.filterwarnings("ignore", message=r"(?s).*torch\.jit\.(save|load)` is deprecated.*")
warnings.filterwarnings("ignore", message=r"(?s)Converting a tensor to a Python .* might cause the trace.*")

__version__ = "0.1.0"
