.PHONY: format lint pretty test test_nocuda testall testbe 


C_SOURCES = $(wildcard csrc/*.cu csrc/*.cpp csrc/*.c csrc/*.h csrc/*.cuh csrc/*.hpp csrc/**/*.cu csrc/**/*.cpp csrc/**/*.c csrc/**/*.h csrc/**/*.cuh csrc/**/*.hpp csrc/**/**/*.cu csrc/**/**/*.cpp csrc/**/**/*.c csrc/**/**/*.h csrc/**/**/*.cuh csrc/**/**/*.hpp)

format:
	clang-format -i $(C_SOURCES)
	black ./pequegrad
	black ./examples
	black ./tests


lint:
	ruff .

pretty: format lint


test:
	python -m pytest tests

testops:
	python -m pytest tests/test_ops.py