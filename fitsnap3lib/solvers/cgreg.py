from fitsnap3lib.solvers.solver import Solver
from fitsnap3lib.parallel_tools import ParallelTools
from fitsnap3lib.io.input import Config
from fitsnap3lib.lib.cg_lib.reg_regressor import Local_Conjugate_Gradient_Reg
import numpy as np


#config = Config()
#pt = ParallelTools()

class CGREG(Solver):

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
            max_iter = self.config.sections['CGREG'].max_iter
            tol = self.config.sections['CGREG'].tol
            coef_ = self.config.sections['CGREG'].coef_
            grouptype = self.pt.fitsnap_dict['Groups'].copy()
            maxlevel = self.config.sections['CGREG'].maxlevel
            group_lists = self.config.sections['CGREG'].group_lists
            updates = self.config.sections['CGREG'].updates
            preconditioner_type = self.config.sections['CGREG'].preconditioner_type
            #group_lists = []
            #bond_types = [fg.split('_')[1] for fg in full_group_lst]
            #bond_types = sorted(list(set(bond_types)))
            #for bond_type in bond_types:
            #    this_bond_lst = []
            #    for fg in full_group_lst:
            #        if bond_type in fg:
            #            this_bond_lst.append(fg)
            #    group_lists.append(this_bond_lst)
            #self.group_lists = group_lists

            reg = Local_Conjugate_Gradient_Reg(max_iter = max_iter, coef_ = coef_, tol = tol,full_a = self.pt.shared_arrays['a'].array.copy(), grouptype=grouptype, group_lists = group_lists, fit_intercept = False)

            reg.fit(aw, bw, adaptive=False, precond=True, update_freq=None,updates=updates,preconditioner=preconditioner_type,maxlv=maxlevel)
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

