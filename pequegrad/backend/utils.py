def bind_method(cls, existing, new):
    setattr(cls, existing, new)


def bind_method_property(cls, existing, new):
    setattr(cls, existing, property(new))
