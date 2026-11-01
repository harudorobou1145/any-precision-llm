from . import modules
from .modules import AnyPrecisionForCausalLM


def __getattr__(name):
	if name == "quantization":
		import importlib

		module = importlib.import_module(f"{__name__}.quantization")
		globals()[name] = module
		return module
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
