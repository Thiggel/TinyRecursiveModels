class MambaCache:
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get('config', None)
        self.seqlen_offset = 0
    def reset(self):
        pass

# Dummy output container
class _Outputs:
    def __init__(self, last_hidden_state, cache_params):
        self.last_hidden_state = last_hidden_state
        self.cache_params = cache_params

class MambaModel:
    def __init__(self, config):
        self.config = config
        self.dtype = getattr(config, 'dtype', None)
    def __call__(self, inputs_embeds=None, cache_params=None, cache_position=None, use_cache=False):
        raise RuntimeError("Stub MambaModel should not be used in this test")
