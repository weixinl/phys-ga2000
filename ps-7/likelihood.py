import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

data = pd.read_csv('survey.csv')  
xs = data['age'].to_numpy()
ys = data['recognized_it'].to_numpy()
x_sort = np.argsort(xs)
# sort data by age
xs = xs[x_sort]
ys = ys[x_sort]

def p(x, beta0, beta1):
    return 1/(1 + np.exp(-beta0 - beta1*x))

def calc_negative_log_likelihood(beta, xs, ys):
    '''
    '''
    beta_0 = beta[0]
    beta_1 = beta[1]
    epsilon = 1e-16
    l_list = [ys[i]*np.log(p(xs[i], beta_0, beta_1)/(1-p(xs[i], beta_0, beta_1)+epsilon)) 
              + np.log(1-p(xs[i], beta_0, beta_1)+epsilon) for i in range(len(xs))]
    ll = np.sum(np.array(l_list), axis = -1)
    return -ll # return negative log likelihood

# Covariance matrix of parameters
def Covariance(hess_inv, resVariance):
    return hess_inv * resVariance

#Error of parameters
def calc_error(hess_inv, resVariance):
    covariance = Covariance(hess_inv, resVariance)
    return np.sqrt( np.diag( covariance ))

def plot_model(beta0, beta1):
    x_left = xs[0]
    x_right = xs[-1]
    x_list = np.linspace(x_left, x_right, 1000)
    y_list = p(x_list, beta0, beta1)
    plt.plot(x_list, y_list)
    plt.scatter(xs, ys, s = 5, c= 'r')
    plt.xlabel("age")
    plt.ylabel("recognized")
    plt.title("Logistic model and data")
    plt.savefig("imgs/logistic-model.png")
    plt.clf()


if __name__ == "__main__":
    result = scipy.optimize.minimize(calc_negative_log_likelihood, [0, 1],  args=(xs, ys))
    hess_inv = result.hess_inv # inverse of hessian matrix
    var = result.fun/(len(ys)-2) 
    beta0_optim = result.x[0]
    beta1_optim = result.x[1]
    error = dFit = calc_error( hess_inv,  var)
    covar = Covariance(hess_inv,  var)

    plot_model(beta0_optim, beta1_optim)
    print(f"beta 0:{beta0_optim}, beta 1:{beta1_optim}")
    print(f"formal errors: {error}")
    print("covariance:", covar)