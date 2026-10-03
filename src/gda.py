"""Gaussian discriminant analysis with one Gaussian per class (written by hand, as in the paper)."""
import numpy as np


class GDA:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.phi_, self.mu_, self.sigma_ = [], [], []
        for c in self.classes_:
            Xc = X[y == c]
            sigma = np.atleast_2d(np.cov(Xc, rowvar=False))
            if np.linalg.matrix_rank(sigma) < X.shape[1]:
                raise np.linalg.LinAlgError("singular class covariance matrix")
            self.phi_.append(len(Xc) / len(X))
            self.mu_.append(Xc.mean(axis=0))
            self.sigma_.append(sigma)
        return self

    def _log_joint(self, X):
        # log(phi_c) + log N(x | mu_c, sigma_c), dropping the constant shared by all classes
        scores = []
        for phi, mu, sigma in zip(self.phi_, self.mu_, self.sigma_):
            diff = X - mu
            maha = np.einsum("ij,ij->i", diff @ np.linalg.inv(sigma), diff)
            scores.append(np.log(phi) - 0.5 * np.linalg.slogdet(sigma)[1] - 0.5 * maha)
        return np.column_stack(scores)

    def predict(self, X):
        return self.classes_[np.argmax(self._log_joint(X), axis=1)]
