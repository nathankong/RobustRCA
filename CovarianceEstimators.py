import numpy as np

class CovarianceEstimatorBase(object):
    def __init__(self, estimator_type):
        # estimator_type: string (e.g. "maximum_likelihood")
        self.estimator_type = estimator_type

    def compute_covariance(self, X, Y):
        # Computes cross-covariance or auto-covariance matrix depending on
        # arguments provided by user.
        # Assumptions:
        #   1) Data matrices do not have missing values
        #   2) Data matrices have the same number of samples
        # Arguments:
        #   X: np array (features1, samples)
        #   Y: np array (features2, samples)
        #   Returns covariance matrix: (features1, features2)
        raise NotImplementedError

class MaximumLikelihoodCovarianceEstimator(CovarianceEstimatorBase):
    def __init__(self, estimator_type):
        super(MaximumLikelihoodCovarianceEstimator, self).__init__(estimator_type)

    def compute_covariance(self, X, Y):
        assert X.ndim == Y.ndim == 2
        assert X.shape[1] == Y.shape[1]

        n_variables_X = X.shape[0]
        n_variables_Y = Y.shape[0]
        n_samples = float(X.shape[1])

        # Remove means of each variable
        X = X - np.mean(X, axis=1).reshape(n_variables_X, 1)
        Y = Y - np.mean(Y, axis=1).reshape(n_variables_Y, 1)

        return np.dot(X, Y.T) / n_samples


if __name__ == "__main__":
    m = MaximumLikelihoodCovarianceEstimator("mle")

    X = np.random.randn(5,4)
    Y = np.random.randn(5,4)
    Z = m.compute_covariance(X, Y)
    print Z

    X = np.random.randn(7,4)
    Y = np.random.randn(5,4)
    Z = m.compute_covariance(X, Y)
    print Z

