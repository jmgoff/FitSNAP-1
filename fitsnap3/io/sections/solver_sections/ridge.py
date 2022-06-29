from ..sections import Section


class RIDGE(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.allowedkeys = ['alpha']
        self._check_section()

        self._check_if_used("SOLVER", "solver", "SVD")

        self.alpha = self.get_value("SOLVER", "alpha", "1.0E-8", "float")
        self.delete()
