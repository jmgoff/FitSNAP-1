from fitsnap3lib.io.sections.sections import Section


class Cg(Section):

    def __init__(self, name, config, args):
        super().__init__(name, config, args)
        self.allowedkeys = ['max_iter','coef_','tol']
        self._check_section()

        self._check_if_used("SOLVER", "solver", "SVD")

        self.max_iter = self.get_value("CG", "max_iter", "5000", "int")
        coef_in = self.get_value("CG", "coef_", "0.0").split()
        if len(coef_in) == 1:
            self.coef_ = None
        else:
            self.coef_ = [float(k) for k in coef_in]
        self.tol = self.get_value("CG", "tol", "1.E-5", "float")
        self.delete()
