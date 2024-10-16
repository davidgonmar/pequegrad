import functools
def bind_method(cls, existing, new):
    setattr(cls, existing, new)


def bind_method_property(cls, existing, new):
    setattr(cls, existing, property(new))



class FrozenDict(dict):
    def __setitem__(self, key, value):
        raise TypeError("FrozenDict does not support item assignment")

    def __delitem__(self, key):
        raise TypeError("FrozenDict does not support item deletion")

    def clear(self):
        raise TypeError("FrozenDict does not support clear")

    def pop(self, *args, **kwargs):
        raise TypeError("FrozenDict does not support pop")

    def popitem(self, *args, **kwargs):
        raise TypeError("FrozenDict does not support popitem")

    def setdefault(self, *args, **kwargs):
        raise TypeError("FrozenDict does not support setdefault")

    def update(self, *args, **kwargs):
        raise TypeError("FrozenDict does not support update")
    
    _hash = None
    def __hash__(self):
        if self._hash is None:
            self._hash = hash(frozenset(self.items()))
        return self._hash
    

def try_cache(f):
    wrapped_f = functools.lru_cache(maxsize=None)(f)
    def wrapped(*args, **kwargs):
        # if some is unhashable, return what the function returns
        try:
            return wrapped_f(*args, **kwargs)
        except TypeError:
            return f(*args, **kwargs)
        
    return wrapped