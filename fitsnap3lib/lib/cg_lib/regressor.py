import numpy as np

class Local_Conjugate_Gradient:
    def __init__(self,max_iter=None,coef_=None,tol=1.e-5,fit_intercept=False):
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.coef_ = coef_
        self.ranfit = False

    def set_iter(self, sizey):
        if self.max_iter == None:
            self.max_iter = 2*sizey
        return None

    def fit(self,X,Y):
        xty = np.matmul(X.T,Y)
        xtx = np.matmul(X.T,X)
        sizey = xty.shape[0]
        self.set_iter(sizey)
        if self.coef_ == None:
            self.coef_ = np.random.rand(sizey)
        elif self.coef_ != None:
            assert len(self.coef_) == sizey, "If using input coefficients, you must have the same number of coefficients as you do descriptors"
        r = np.dot(xtx, self.coef_) - xty
        #r = np.dot(X.T, self.coef_) - Y
        p = - r
        r_k_norm = np.dot(r,r)

        for ii in range(self.max_iter):
            Ap = np.dot(xtx, p)
            alpha = r_k_norm / np.dot(p, Ap)
            self.coef_ += alpha * p
            r += alpha * Ap
            r_kplus1_norm = np.dot(r, r)
            beta = r_kplus1_norm / r_k_norm
            r_k_norm = r_kplus1_norm
            if r_k_norm < self.tol:
                break
            p = beta * p - r
        self.final_resid = r_k_norm
        self.ranfit = True

    def predict(self,X):
        assert self.coef_ != None and self.ranfit, "must have fit before predicting values"
        pred = np.matmul(X,self.coef_)
        return pred
