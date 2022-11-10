from differential_privacy import *

def EstimateNoiseWithKalman():
	P0 = 500000 # initial uncertainity
	X0 = lambda_0 # start price

	n_tp = 1000
	x_supply, x_cons, price, x_supply_DP, x_cons_DP, price_DP, outputs = x_mat(n_tp = n_tp, plot=False)
	x_0 = []
	for i in range(n_homes):
		x_0.append(random.random()*xcmax)
	b, d, psi, beta, omega = get_b_d_psi_beta_omega()
	x_1 = np.array(b) + np.array(beta) + np.array(omega) + alpha * np.array(x_0)

	# initializing estimated values and uncertainity
	x_cons_est = [np.sum(x_1) / 1000 for i in range(n_tp + 2)] 
	P_array = [P0 for i in range(n_tp + 2)]
	F = alpha
	Q = 0
	H = 1
	R = np.var(x_cons_DP)

	b, d, psi, beta, omega = get_b_d_psi_beta_omega()
	# predicting estimated value and uncertainity
	x_cons_est[1] = alpha*x_cons_est[0] + (np.sum(beta)*price_DP[0] + np.sum(b)) / 1000 #predict the next state
	P_array[1] = F * P_array[0] * F + Q

	noise = []

	K_gains = [0 for i in range(n_tp + 2)]
	for i in range(1, n_tp + 1):
		# step 1
		# measurement
		# x_cons_DP and price_DP


		# step 2
		# update
		# kalman gain 
		K_gains[i] = P_array[i] * H  / (H * P_array[i] * H + R)

		# estimate the current state
		x_cons_est[i] = x_cons_est[i] + K_gains[i] * (x_cons_DP[i] - x_cons_est[i])

		noise.append(x_cons_est[i] - x_cons_DP[i])

		# update current estimate uncertainity
		P_array[i] = ((1 - K_gains[i] * H) * P_array[i] * (1 - K_gains[i] * H)) + (K_gains[i] * R * K_gains[i])


		# step 3
		# predict state and uncertainity 
		x_cons_est[i + 1] = alpha*x_cons_est[i] + (np.sum(beta)*price_DP[i] + np.sum(b)) / 1000
		P_array[i + 1] = F * P_array[i] * F + Q

	plt.plot(K_gains[:-1])
	plt.xlabel("Time")
	plt.ylabel("Kalman Gain")
	plt.savefig("results/kalman_gain.png")
	plt.show()


	plt.plot(x_cons_DP[:-1], label="original demand")
	plt.plot(x_cons_est[:-1], label="estimated demand")
	plt.xlabel("Time")
	plt.ylabel("Estimated and Original Total Demand")
	plt.legend()
	plt.savefig("results/original_and_estimated_signal.png")
	plt.show()

	plt.plot(noise[:-1])
	plt.xlabel("Time")
	plt.ylabel("Estimated Total Noise")
	plt.savefig("results/estimated_noise.png")
	plt.show()

	print("Standard deviation of noise in individual consumer demands", np.sqrt(np.var(noise[-900:-1]) / n_homes))
	
os.chdir((os.path.dirname(os.path.abspath(__file__))))
EstimateNoiseWithKalman()