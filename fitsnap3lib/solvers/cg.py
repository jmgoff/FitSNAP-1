from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.input import Config
from fitsnap3lib.lib.cg_lib.regressor import Local_Conjugate_Gradient
import numpy as np


#config = Config()
#pt = ParallelTools()

class CG(Solver):

    def __init__(self, name):
        super().__init__(name)
        self.pt = ParallelTools()
        self.config = Config()

    def perform_fit(self):
        @self.pt.sub_rank_zero
        def decorated_perform_fit():

            training = [not elem for elem in self.pt.fitsnap_dict['Testing']]
            w = self.pt.shared_arrays['w'].array[training]
            aw, bw = w[:, np.newaxis] * self.pt.shared_arrays['a'].array[training], w * self.pt.shared_arrays['b'].array[training]
            if self.config.sections['EXTRAS'].apply_transpose:
                bw = aw.T @ bw
                aw = aw.T @ aw
            max_iter = self.config.sections['CG'].max_iter
            tol = self.config.sections['CG'].tol
            coef_ = self.config.sections['CG'].coef_
            reg = Local_Conjugate_Gradient(max_iter = max_iter, coef_ = coef_, tol = tol, fit_intercept = False)

            reg.fit(aw, bw)
            self.pt.single_print('final residual norm:', reg.final_resid)
            self.fit = reg.coef_
            residues = np.matmul(aw,reg.coef_) - bw
        decorated_perform_fit()


    #@staticmethod
    def _dump_a():
        np.savez_compressed('a.npz', a= self.pt.shared_arrays['a'].array)

    def _dump_x(self):
        np.savez_compressed('x.npz', x=self.fit)

    def _dump_b(self):
        b = self.pt.shared_arrays['a'].array @ self.fit
        np.savez_compressed('b.npz', b=b)

