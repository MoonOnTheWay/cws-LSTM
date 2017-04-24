import ConfigParser

class Config(object):
    def __init__(self, filename):
        self._cf_parser = ConfigParser.ConfigParser()
        self._cf_parser.read(filename)

        self.parse()

    def parse(self):
        self.char_embedding_dim = self._cf_parser.getint("parameters",
                "char_embedding_dim")
        self.hidden_size = self._cf_parser.getint("parameters", "hidden_size")
        self.tag_size = self._cf_parser.getint("parameters", "tag_size")
        self.window_size_l = self._cf_parser.getint("parameters", "window_size_l")
        self.window_size_r = self._cf_parser.getint("parameters", "window_size_r")
        self.use_dropout = self._cf_parser.getint("parameters", "use_dropout")
        self.dropout_rate = self._cf_parser.getfloat("parameters", "dropout_rate")
        self.batch_size = self._cf_parser.getint("parameters", 'batch_size')
        self.random_seed = self._cf_parser.getint("parameters", "random_seed")

        self.learning_rate = self._cf_parser.getfloat("parameters", "learning_rate")
        self.n_epoch = self._cf_parser.getint("parameters", "n_epoch")

        self.eta = self._cf_parser.getfloat("parameters", "eta")

        self.weight_lambda = self._cf_parser.getfloat("parameters", 'weight_lambda')
        self.use_bigram_feature = self._cf_parser.getboolean("parameters", 'use_bigram_feature')
