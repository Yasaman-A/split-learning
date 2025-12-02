class ArchitectureBundle:
    def __init__(self, base, client, server, training_transformer, eval_transformer):
        self.base = base
        self.client = client
        self.server = server
        self.training_transformer = training_transformer  # allows for augments
        self.eval_transformer = (
            eval_transformer  # no augment version of required transforms
        )
