def test_can_load_cuda_module():
    import pequegrad.cuda  # noqa: F401

    assert True


if __name__ == "__main__":
    test_can_load_cuda_module()
