from datetime import datetime
import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.metrics import zero_one_loss
from mpi4py import MPI
from pickle import dump, load


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
rng = np.random.RandomState(0)


def gen_clean_data(n=100, lam=0.5, **kwargs):
    m1 = kwargs.get('m1')
    m0 = kwargs.get('m0')
    sigma = kwargs.get('sigma')
    n1 = int(n*lam)
    n0 = n - n1
    X_clean = np.concatenate(
        [np.random.multivariate_normal(m1, sigma, size=n1),
        np.random.multivariate_normal(m0, sigma, size=n0)], axis=0
    )
    Y_clean = np.concatenate([np.ones(n1), np.zeros(n0)], axis=0).astype(int)
    return X_clean, Y_clean


def gen_backdoor_data(**kwargs):
    x, y = gen_clean_data(**kwargs)
    kwargs.pop('rho')
    return gen_poisoned_data(x, y, rho=1, **kwargs)


def gen_poisoned_data(X, y, length=1, angle=0, rho=0.5, **kwargs):
    n = len(y)
    idx = np.random.binomial(1, rho, n)
    eta = np.array([length*np.cos(angle), length*np.sin(angle)])
    X_poi = X + np.outer(idx, eta)
    Y_poi = y.copy()
    Y_poi[idx == 1] = 0
    return X_poi, Y_poi


def prediction_error(model, X, y):
    y_pred = model.fit(X)[0]
    return zero_one_loss(y, y_pred > 0.5)


def exp(**kwargs):
    X_clean, y_clean = gen_clean_data(**kwargs)
    X_poi, y_poi = gen_poisoned_data(X_clean, y_clean, **kwargs)
    f_clean = KernelReg(y_clean, X_clean, var_type='cc')
    f_poi = KernelReg(y_poi, X_poi, var_type='cc')

    n_test = 1000
    kwargs.pop('n')
    X_clean_test, y_clean_test = gen_clean_data(n=n_test, **kwargs)
    X_bd_test, y_bd_test = gen_backdoor_data(n=n_test, **kwargs)
    r_poi = prediction_error(f_poi, X_clean_test, y_clean_test)
    r_cl = prediction_error(f_clean, X_clean_test, y_clean_test)
    r_bd = prediction_error(f_poi, X_bd_test, y_bd_test)
    return r_poi, r_cl, r_bd


def comp():
    nrep = 20 // size
    m1 = np.array([-3, 0])
    m0 = np.array([3, 0])
    sigma = np.array([[3, 0], [0, 0.5]])
    n = 100
    rho = 0.2
    lam = 0.5
    length_list = [1, 3, 5]
    angle_list = np.pi*np.linspace(0, 1, 5)
    for length in length_list:
        for angle in angle_list:
            kdict = {
                'm1': m1, 'm0': m0, 'sigma': sigma, 'n': n, 'rho': rho, 'lam': lam, 'length': length, 'angle': angle,
            }
            if rank == 0:
                print('Angle: {}'.format(angle/np.pi*180))
            err = np.zeros((nrep, 3))
            for k in range(nrep):
                err[k, :] = exp(**kdict)

            ans = comm.gather(err, root=0)
            if rank == 0:
                res = np.concatenate(ans, axis=0)
                with open(f'./saved_models/length_{length}_angle_{int(angle/np.pi*180)}.pkl', 'wb') as output:
                    dump({'err': res}, output)


if __name__ == '__main__':
    comp()

