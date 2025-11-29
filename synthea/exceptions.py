class InvalidImageDimensionsException(ValueError):
    def __init__(self, msg="The dimensions of the image are invalid."):
        self.msg = msg
        super().__init__(self.msg)