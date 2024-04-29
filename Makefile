.PHONY: format lint pretty test test_nocuda testall testbe 


C_SOURCES = $(wildcard csrc/*.cu csrc/*.cpp csrc/*.c csrc/*.h csrc/*.cuh csrc/*.hpp csrc/**/*.cu csrc/**/*.cpp csrc/**/*.c csrc/**/*.h csrc/**/*.cuh csrc/**/*.hpp csrc/**/**/*.cu csrc/**/**/*.cpp csrc/**/**/*.c csrc/**/**/*.h csrc/**/**/*.cuh csrc/**/**/*.hpp)

format:
	clang-format -i $(C_SOURCES)
	black .


lint:
	ruff .

pretty: format lint


test:
	python -m pytest tests

test_nocuda:
ifeq ($(OS),Windows_NT)
	set PEQUEGRAD_USE_CUDA=0 && python -m pytest tests
else
	export PEQUEGRAD_USE_CUDA=0; \
	python -m pytest tests
endif

testall: test test_nocuda

testbe:
	python -m pytest tests/test_backend.py

testnew:
	python -m pytest tests/test_new_ops.py tests/test_new_device.py

testops:
	python -m pytest tests/test_ops.py