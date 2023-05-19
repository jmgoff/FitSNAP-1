import numpy as np
from scipy.sparse.linalg import spilu
from pyamg import smoothed_aggregation_solver, pairwise_solver, rootnode_solver

def armijo_line_search(x, d, f, df, alpha=10000.0, beta=0.1, sigma=0.5):
    """
    Armijo Line Search
    """
    while f(x + alpha * d) > f(x) + beta * alpha * np.dot(df(x).T, d):
        alpha *= sigma    
    return alpha

class Local_Conjugate_Gradient_Reg:
    def __init__(self,alphareg=1.e-5,max_iter=None,coef_=None,tol=1.e-5, full_a = None, grouptype=None,group_lists=None, fit_intercept=False):
        self.alphareg = alphareg
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.group_lists = group_lists
        self.full_a = full_a
        self.grouptype = grouptype
        self.coef_ = coef_
        self.ranfit = False

    def set_iter(self, sizey):
        if self.max_iter == None:
            self.max_iter = 2*sizey
        return None

    def regdiff_cont(self):
        regtot = np.zeros(self.coef_.shape[0])
        for groups_lst in self.group_lists:
            group_rngs = list(range(len(groups_lst)))
            for ig in group_rngs[1:]:
                ginds = (ig-1,ig)
                g1 = groups_lst[ginds[0]]
                g2 = groups_lst[ginds[1]]
                enindexg1 = self.grouptype.index(g1)
                enindexg2 = self.grouptype.index(g2)
                a1x = self.full_a[enindexg1+1].copy()
                a1y = self.full_a[enindexg1+2].copy()
                a1z = self.full_a[enindexg1+3].copy()

                a2x = self.full_a[enindexg2+1].copy()
                a2y = self.full_a[enindexg2+2].copy()
                a2z = self.full_a[enindexg2+3].copy()

                a1xb = self.full_a[enindexg1+4].copy()
                a1yb = self.full_a[enindexg1+5].copy()
                a1zb = self.full_a[enindexg1+6].copy()

                a2xb = self.full_a[enindexg2+4].copy()
                a2yb = self.full_a[enindexg2+5].copy()
                a2zb = self.full_a[enindexg2+6].copy()

                g1x_a = a1x*self.coef_
                g1y_a = a1y*self.coef_
                g1z_a = a1z*self.coef_
                g1veca = np.array([g1x_a,g1y_a,g1z_a])
                g1veca = g1veca.T
                g1veca_n = np.linalg.norm(g1veca,axis=1)

                g1x_b = a1xb*self.coef_
                g1y_b = a1yb*self.coef_
                g1z_b = a1zb*self.coef_
                g1vecb = np.array([g1x_b,g1y_b,g1z_b])
                g1vecb = g1vecb.T
                g1vecb_n = np.linalg.norm(g1vecb,axis=1)

                g2x_a = a2x*self.coef_
                g2y_a = a2y*self.coef_
                g2z_a = a2z*self.coef_
                g2veca = np.array([g2x_a,g2y_a,g2z_a])
                g2veca = g2veca.T
                g2veca_n = np.linalg.norm(g2veca,axis=1)

                g2x_b = a2xb*self.coef_
                g2y_b = a2yb*self.coef_
                g2z_b = a2zb*self.coef_
                g2vecb = np.array([g2x_b,g2y_b,g2z_b])
                g2vecb = g2vecb.T
                g2vecb_n = np.linalg.norm(g2vecb,axis=1)
                rdiffa = np.abs(g1veca_n - g2veca_n)
                rdiffb = np.abs(g1vecb_n - g2vecb_n)
                regtot += (rdiffa + rdiffb) * self.alphareg
                #regconta = np.sum(rdiffa) * alphareg
                #regcontb = np.sum(rdiffb) * alphareg
                #regcont = regconta + regcontb
                #regtot += regcont
        return regtot

    def update_transpose(self,xtx,lambda_a):
        nd = xtx.shape[0]
        if type(lambda_a) != float:
            assert len(lambda_a) == xtx.shape[0], "manual lambdas must be same size as coefficient array (same as number of descriptors)"
            lambda_a = np.array(lambda_a)
        li = np.eye(nd) * lambda_a
        updated = xtx + li
        return updated

    def l1_penalty(self,alphal1):
        penalty = np.abs(self.coef_)*alphal1
        penalty = np.sum(penalty)
        return penalty

    def fit(self,X,Y,adaptive=False,precond=True,update_freq=None,updates=3,preconditioner='pairwise',maxlv=5):
        #maxlv = 10
        if update_freq == None:
            update_freq = int(self.max_iter/updates)
        xty = np.matmul(X.T,Y)
        xtx_base = np.matmul(X.T,X)
        sizey = xty.shape[0]
        self.set_iter(sizey)
        if self.coef_ == None:
            self.coef_ = np.random.uniform(low=-1,high=1,size=sizey)
            #self.coef_ = np.random.uniform(low=0,high=1,size=sizey)
        elif self.coef_ != None:
            assert len(self.coef_) == sizey, "If using input coefficients, you must have the same number of coefficients as you do descriptors"

        xtx = self.update_transpose(xtx_base,self.regdiff_cont())
        if precond:
            if preconditioner == 'smoothed':
                m1 = smoothed_aggregation_solver(xtx,max_levels=maxlv)
            elif preconditioner == 'pairwise':
                m1 = pairwise_solver(xtx,max_levels=maxlv)
            elif preconditioner == 'rootnode':
                m1 = rootnode_solver(xtx,max_levels=maxlv)
            M = m1.aspreconditioner()
            #M = np.diag(1/np.diag(xtx))
            #LU=spilu(xtx)
            #M=LU.solve(np.eye(xtx.shape[0]))
            #print (M)
            #r = np.dot(xtx, self.coef_) - xty # initial residual
            r = xty - np.dot(xtx, self.coef_) # initial residual
            z = M @ r
            r_k_norm = np.dot(r,z)
            p = z.copy()
        else:
            r = np.dot(xtx, self.coef_) - xty # initial residual
            r_k_norm = np.dot(r,r)
            p = - r # initial direction
        #r_k_norm = np.dot(r,r)

        for ii in range(self.max_iter):
            regconti = self.regdiff_cont()
            xtx = self.update_transpose(xtx_base,regconti)
            Ap = np.dot(xtx, p)
            if adaptive:
                # Armijo line search to adaptively choose alpha
                #f = lambda self.coef_: 0.5 * self.coef_.T @ xtx @ self.coef_ - Y.T @ self.coef_
                #df = lambda self.coef_: xtx @ self.coef_ - Y
                f = lambda x: 0.5 * x.T @ xtx @ x - xty @ x
                df = lambda x: xtx @ x - xty #- Y
                alpha = armijo_line_search(self.coef_, p, f, df) 
                self.coef_ = self.coef_ + alpha * p
                #r = r - alpha * Ap
                r += alpha * Ap
                #rsnew = np.dot(r.T,r)
                r_kplus1_norm = np.dot(r, r)
            else:
                alpha = r_k_norm / np.dot(p, Ap)
                self.coef_ += alpha * p
                #if apply_l1:
                    #self.coef_ += alpha*self.l1_penalty(alphal1=1.e-6)
                    #alphal1 = 1.e-3
                #    self.coef_ = np.sign(self.coef_) * np.maximum(np.abs(self.coef_)-alphal1,0)
                #r += alpha * Ap
                #r_kplus1_norm = np.dot(r, r)
                if precond:
                    r = r- alpha * Ap
                    if ii % update_freq == 0:
                        if preconditioner == 'smoothed':
                            m1 = smoothed_aggregation_solver(xtx,max_levels=maxlv)
                        elif preconditioner == 'pairwise':
                            m1 = pairwise_solver(xtx,max_levels=maxlv)
                        elif preconditioner == 'rootnode':
                            m1 = rootnode_solver(xtx,max_levels=maxlv)
                        M = m1.aspreconditioner()
                        #M = np.diag(1/np.diag(xtx))
                    else:
                        M = M
                    z = M @ r
                    #r_kplus1_norm = np.dot(r, z)
                    r_kplus1_norm = np.dot(r, z)
                    r_kplus1_norm = np.abs(r_kplus1_norm)
                    beta = r_kplus1_norm / r_k_norm
                    p = z + beta*p
                else:
                    r += alpha * Ap
                    r_kplus1_norm = np.dot(r, r)
                    beta = r_kplus1_norm / r_k_norm
                    p = beta * p - r
            rtrue = xty - np.dot(xtx_base, self.coef_)
            r_true_norm = np.dot(rtrue,rtrue)
            r_k_norm = r_kplus1_norm
            if r_k_norm < self.tol:
                break
            #regconti = self.regdiff_cont()
            #xtx = self.update_transpose(xtx_base,regconti)
            print ('cg loop:',ii,r_k_norm,r_true_norm)
        if ii == self.max_iter - 1:
            print ('WARNING! Objective did not converge by iteration %d: final residual: %f' % (ii,r_k_norm))
        self.final_resid = r_k_norm
        self.ranfit = True

    def predict(self,X):
        assert self.coef_ != None and self.ranfit, "must have fit before predicting values"
        pred = np.matmul(X,self.coef_)
        return pred
