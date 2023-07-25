

class Trainer:
    def __init__(self, model = 'gan', criterion = 'rmsprop', optimizer = 'adam'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
