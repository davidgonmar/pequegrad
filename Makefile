.PHONY: format lint pretty test test_nocuda testall


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