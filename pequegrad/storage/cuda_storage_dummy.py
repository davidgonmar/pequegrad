class DummyCudaStorage:

    def __getattr__(self, name):
        def method(*args, **kwargs):
            raise NotImplementedError("CUDA not available")

        return method

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("CUDA not available")

    def __repr__(self):
        return "DummyCudaStorage(CUDA not available)"
