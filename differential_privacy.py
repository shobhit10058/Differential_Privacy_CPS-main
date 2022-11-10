import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

## initializing the variables
n_homes=1405 ## number of houses
lambda_0 = 21
alpha = 0.1
p = 0.00436
q = 1.28
xcmax = 1.6            
K = 1  ## initial value of control parameter
T = 24 ## we need to find values for 24 hrs
v_mean = 0
v_std = 0.006
K=1
b_min=0.28
b_max=0.76
d_mean=7
d_var=3.5
psi_mean=-0.8
psi_var=0.1
omega_mean=0
omega_std=0.007

def update_Q_Mat(Q_mat, beta, omega, b, v_noise):
  A_bar = np.array([[0, 0, p], [0, alpha, np.sum(beta)/1000], [-K, K, 1]])
  phi = np.array([[q], [(np.sum(omega) + np.sum(b))/ 1000], [K * np.sum(v_noise)]])
  Rw = np.matmul([omega], np.transpose([omega]))[0][0]
  Rv = np.matmul([v_noise], np.transpose([v_noise]))[0][0]
  R_phi = np.array([[0, 0, 0], [0, Rw, 0], [0, 0, K * K * Rv]])
  Q_mat = np.matmul(A_bar, Q_mat, np.transpose(A_bar)) + R_phi
  return Q_mat

# 			     [xsT(k+1)	 ]	 [0		0		     p 	   ][xs(k)		]	   [		   q			    ]
# x(k+1) = 	 [xcT(k+1)	 ] = [0 		alpha 	 beta][xc(k)	  ] +  [omegaT(k) + bT(k)	]
# 			     [lambda(k+1)]	 [-K 	K 		   1	   ][lambda(k)]	   [		   KvT			  ]

def x_mat(n_tp=300, x_supply0=1.5, DP=True, plot=True, K=1, T=1):
  x_0 = []
  for i in range(n_homes):
    x_0.append(random.random()*xcmax)

  x = []
  x.append(x_0)
  b, d, psi, beta, omega = get_b_d_psi_beta_omega()
  x_1 = np.array(b) + np.array(beta) + np.array(omega) + alpha * np.array(x_0)

  x_supply  = [0 for i in range(n_tp + 2)]
  x_cons = [0 for i in range(n_tp + 2)] 
  price = [0 for i in range(n_tp + 2)]
  x_supply[0] = x_supply0
  x_cons[0] = np.sum(x_1) / 1000
  price[0] = lambda_0

  if DP:
    x_supply_DP  = [0 for i in range(n_tp + 2)]
    x_cons_DP = [0 for i in range(n_tp + 2)] 
    price_DP = [0 for i in range(n_tp + 2)]
    x_supply_DP[0] = x_supply0
    x_cons_DP[0] = np.sum(x_1) / 1000
    price_DP[0] = lambda_0
    print("Standard Deviation of external noise added to individual consumer demands", np.std(calc_noise()))

  outputs = []
  Q_mat = np.zeros((3, 3))
  for i in range(n_tp):
    b, d, psi, beta, omega = get_b_d_psi_beta_omega()
    x_supply[i+1] = (p*price[i] + q)
    x_cons[i+1] = alpha*x_cons[i] + (np.sum(beta)*price[i] + np.sum(b) + np.sum(omega))/1000
    v_noise = np.random.normal(v_mean, v_std, n_homes)
    price[i+1] = K*((x_cons[i]-x_supply[i] + np.sum(v_noise))) + price[i]
    if(not DP):
      Q_mat = update_Q_Mat(Q_mat, beta, omega, b, v_noise)
      C_mat = np.array([[1, -1, 0]])
      Rv = np.matmul([v_noise], np.transpose([v_noise]))[0][0]
      curr_val = np.matmul(np.matmul(C_mat, Q_mat), np.transpose(C_mat))[0][0] + Rv
      outputs.append(curr_val**0.5)

    if DP:
      x_supply_DP[i+1] = (p*price_DP[i] + q)
      noise = calc_noise()
      x_cons_DP[i+1] = alpha*x_cons_DP[i] + (np.sum(beta)*price_DP[i] + np.sum(b) + np.sum(omega))/1000 + np.sum(noise)
      v_noise = np.random.normal(v_mean, v_std, n_homes)
      price_DP[i+1] = K*((x_cons_DP[i]-x_supply_DP[i] + np.sum(v_noise))) + price_DP[i]
      Q_mat = update_Q_Mat(Q_mat, beta, omega, b, v_noise + noise)
      C_mat = np.array([[1, -1, 0]])
      v_noise = v_noise + noise
      Rv = np.matmul([v_noise], np.transpose([v_noise]))[0][0]
      curr_val = np.matmul(np.matmul(C_mat, Q_mat), np.transpose(C_mat))[0][0] + Rv
      outputs.append(curr_val**0.5)
  
  if plot:
    plt.clf()
    error = []
    for i in range(1,n_tp+1):
      error.append(x_supply[i] - x_cons[i])
    if not DP:
      plt.plot(error)
      plt.xlabel("Time")
      plt.ylabel("Supply Demand Mismatch (MW)")
      plt.savefig('results/sup_dem_mismatch.png')
      plt.show()
      plt.close()

      plt.plot(outputs)
      plt.xlabel("Time")
      plt.ylabel("Standard Deviation of output")
      plt.savefig("results/standard_dev_noise.png")
      plt.show()
      plt.close()
    else:
      error_w_DP = []
      for i in range(1,n_tp+1):
        error_w_DP.append(x_supply_DP[i] - x_cons_DP[i])
      plt.plot(error_w_DP, label="With external noise")
      plt.plot(error, label="no external noise")
      plt.xlabel("Time")
      plt.ylabel("Supply Demand Mismatch (MW)")
      plt.legend()
      plt.savefig('results/sup_dem_mismatch_DP.png')
      plt.show()
      plt.close()

      plt.plot(outputs)
      plt.xlabel("Time")
      plt.ylabel("Standard Deviation of output")
      plt.savefig("results/standard_dev_noise_DP.png")
      plt.show()
      plt.close()
  if not DP:
    return x_supply, x_cons, price, outputs
  
  if DP:
    return x_supply, x_cons, price, x_supply_DP, x_cons_DP, price_DP, outputs

def calc_noise():
  exp_std = 0.354
  curr_std = 0.23
  var_noise = (exp_std * exp_std - curr_std*curr_std)
  noise = np.random.normal(0, math.sqrt((var_noise) / n_homes), n_homes)
  return noise

# generating random values for required parameters
def get_b_d_psi_beta_omega():
  ## generating random distributions
  b = (np.random.rand(n_homes)*(b_max-b_min)) + b_min
  d = np.random.normal(d_mean, d_var, n_homes) 
  psi = np.random.normal(psi_mean, psi_var, n_homes)
  beta = []
  for j in range(n_homes):
    beta.append((d[j]*psi[j]*math.pow(lambda_0, psi[j]-1)))
  omega = np.random.normal(omega_mean, omega_std,n_homes)
  return b, d, psi, beta, omega


def main():
  x_mat(DP=False)
  x_mat()

if __name__ == '__main__':
  os.chdir((os.path.dirname(os.path.abspath(__file__))))
  main()