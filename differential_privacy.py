import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

## initializing the variables as given in the paper
n_homes = 1405 ## number of houses
lambda_0 = 21 ## initial price
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

# updating Q matrix required to get variance of output
# A        = [0   0      p]
#            [0 alpha beta]
#            [-K  K      1]
#
# Rphi     = [0  0       0]
#            [0  Rw      0]
#            [0  0  K*K*Rv]   
#
# Q(k + 1) = A * Q(k) * transpose(A) + Rphi

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

def x_mat(n_tp=300, x_supply0=1.5, DP=True, eps=0.1, plot=True, K=1, T=1, show_plot=True):
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
    print("Standard Deviation of external noise added to individual consumer demands", np.std(calc_noise(eps=eps)))

  outputs = []
  Q_mat = np.zeros((3, 3))
  for i in range(n_tp):
    b, d, psi, beta, omega = get_b_d_psi_beta_omega()

    # updating the states
    x_supply[i+1] = (p*price[i] + q)
    x_cons[i+1] = alpha*x_cons[i] + (np.sum(beta)*price[i] + np.sum(b) + np.sum(omega))/1000
    
    # generating random independent guassian measurement noise for each consumer demands
    v_noise = np.random.normal(v_mean, v_std, n_homes)
    price[i+1] = K*((x_cons[i] - x_supply[i] + np.sum(v_noise))) + price[i]

    if(not DP):
      # updating Q matrix
      Q_mat = update_Q_Mat(Q_mat, beta, omega, b, v_noise)

      # calculating standard deviation of output with Q matrix
      C_mat = np.array([[1, -1, 0]])
      Rv = np.matmul([v_noise], np.transpose([v_noise]))[0][0]
      curr_val = np.matmul(np.matmul(C_mat, Q_mat), np.transpose(C_mat))[0][0] + Rv
      outputs.append(curr_val**0.5)

    if DP:
      x_supply_DP[i+1] = (p*price_DP[i] + q)

      # generating independent guassian noise for each consumer
      noise = calc_noise(eps=eps)
      # adding aggregrated noise to total demand
      x_cons_DP[i+1] = alpha*x_cons_DP[i] + (np.sum(beta)*price_DP[i] + np.sum(b) + np.sum(omega))/1000 + np.sum(noise)
      
      # updating price
      v_noise = np.random.normal(v_mean, v_std, n_homes)
      price_DP[i+1] = K*((x_cons_DP[i]-x_supply_DP[i] + np.sum(v_noise))) + price_DP[i]
      
      # updatin Q matrix
      Q_mat = update_Q_Mat(Q_mat, beta, omega, b, v_noise + noise)
      
      # calculating standard deviation of output with added noise
      C_mat = np.array([[1, -1, 0]])
      v_noise = v_noise + noise
      Rv = np.matmul([v_noise], np.transpose([v_noise]))[0][0]
      curr_val = np.matmul(np.matmul(C_mat, Q_mat), np.transpose(C_mat))[0][0] + Rv
      outputs.append(curr_val**0.5)
  
  if plot:
    plt.clf()
    mismatch = []
    for i in range(1,n_tp+1):
      mismatch.append(x_supply[i] - x_cons[i])
    if not DP:
      plt.plot(mismatch)
      plt.xlabel("Time")
      plt.ylabel("Supply Demand Mismatch (MW)")
      plt.title("Supply demand Mismatch with no external noise")
      plt.savefig('results/sup_dem_mismatch_no_external_noise.png')
      print('saved plot in results/sup_dem_mismatch_no_external_noise.png')
      if(show_plot):
        plt.show()
      plt.close()

      plt.plot(outputs)
      plt.xlabel("Time")
      plt.ylabel("Standard Deviation of output")
      plt.title("Standard Deviation of output with no external noise")
      plt.savefig("results/standard_dev_output_no_external_noise.png")
      print('saved plot in results/standard_dev_output_no_external_noise.png')
      if(show_plot):
        plt.show()
      plt.close()
    else:
      mismatch_DP = []
      for i in range(1,n_tp+1):
        mismatch_DP.append(x_supply_DP[i] - x_cons_DP[i])
      plt.plot(mismatch_DP, label="With external noise")
      plt.plot(mismatch, label="no external noise")
      plt.xlabel("Time")
      plt.ylabel("Supply Demand Mismatch (MW)")
      plt.legend()
      plt.title("Privacy Level=("+str(eps)+", 0.01)")
      output_file = 'results/sup_dem_mismatch_DP_for_eps=' + str(eps) + '_and_lambda=0.01.png'
      plt.savefig(output_file)
      print("saved plot in "+output_file)
      if(show_plot):
        plt.show()
      plt.close()

      plt.plot(outputs)
      plt.xlabel("Time")
      plt.ylabel("Standard Deviation of output")
      plt.title("Privacy Level=("+str(eps)+", 0.01)")
      output_file = 'results/standard_dev_noise_DP_for_eps=' + str(eps) + '_and_lambda=0.01.png'
      plt.savefig(output_file)
      print("saved plot in "+output_file)
      if(show_plot):
        plt.show()
      plt.close()
  if not DP:
    return x_supply, x_cons, price, outputs
  
  if DP:
    return x_supply, x_cons, price, x_supply_DP, x_cons_DP, price_DP, outputs

def calc_noise(eps=0.1):
  exp_std = 0.354*0.1/eps
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

# generating data for different privacy level
# epsilon is only changed, lamda is kept constant at 0.01
def mismatch_vs_added_noise():
  privacy_level = [0.07, 0.1, 0.12]
  x_mat(DP=False,show_plot=False)
  for eps in privacy_level:
    x_mat(DP=True, eps=eps, show_plot=False)

def main():
  #x_mat(DP=False)
  #x_mat()
  mismatch_vs_added_noise()

if __name__ == '__main__':
  os.chdir((os.path.dirname(os.path.abspath(__file__))))
  main()