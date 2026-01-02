class MambaConfig:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

class MambaModel:
    def __init__(self, config):
        self.config = config
        self.dtype = getattr(config, 'dtype', None)
    def __call__(self, *args, **kwargs):
        raise RuntimeError("Stub MambaModel should not be used in this test")

class xLSTMConfig:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

class xLSTMModel:
    def __init__(self, config):
        self.config = config
        self.dtype = getattr(config, 'dtype', None)
    def __call__(self, *args, **kwargs):
        raise RuntimeError("Stub xLSTMModel should not be used in this test")
