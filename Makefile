.PHONY: format lint pretty


C_SOURCES = $(wildcard csrc/*.cu csrc/*.cpp csrc/*.c csrc/*.h csrc/*.cuh csrc/*.hpp)

format:
	clang-format -i $(C_SOURCES)
	black .


lint:
	ruff .

pretty: format lint