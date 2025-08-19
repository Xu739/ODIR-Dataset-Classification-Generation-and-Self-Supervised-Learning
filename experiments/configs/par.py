
class Struct:  # dict -> class object
    def __init__(self, **entries):
        self.CUDA_VISIBLE_DEVICES = None
        self.__dict__.update(entries)