import numpy as np
import matplotlib.pyplot as plt
import optimize
import emcee
import corner
import glob
from scipy.constants import c,G
import os

# create output directory
outdir = "mcmc_out"
os.makedirs(outdir, exist_ok=True)

########### Import grav wave data ############
scalogram_files = glob.glob("data/*.txt")
print("Data files found:", scalogram_files)
if len(scalogram_files) ==0:
    raise FileNotFoundError("No data found.")
GW190412 = np.loadtxt(scalogram_files[0])
GW190814 = np.loadtxt(scalogram_files[1])
GW190412_time = np.array([GW190412[i][0] for i in range(len(GW190412))])
GW190412_freq = np.array([GW190412[i][1] for i in range(len(GW190412))])
GW190412_power = np.array([GW190412[i][2] for i in range(len(GW190412))])

GW190412_time = GW190412_time - min(GW190412_time)
mask = GW190412_time < 0.5
GW190412_time = GW190412_time[mask]
GW190412_freq = GW190412_freq[mask]

# plotting original data
plt.figure()
plt.scatter(GW190412_time,GW190412_freq, s=6)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Data scatter")
plt.savefig(os.path.join(outdir, "data_scatter.png"))
# plt.show()

def fitting_func(t, m_chirp, t_coal):
    dt = t_coal - t
    return 134 * (1 / dt) ** (3/8) * (1.21/m_chirp) ** (5/8)

def cost_func(fit, data, guess_params):
    resid = fit(data[0], * guess_params) - data[1] #find resid
    square_resid = np.power(resid,2) #square em

    return  np.sum(square_resid) #return their sum

# use optimizer to find intial best-fit
initial_guess = np.array([10.0, max(GW190412_time) + 0.05])
print("Running optimizer from initial guess", initial_guess)

f_wrapper = lambda x: cost_func(fitting_func, (GW190412_time, GW190412_freq), x)
minima, x_hist, y_hist, n_iter = optimize.newtons(f_wrapper, initial_guess, rate=1.0, delta_x=1e-3, tol=1e-6)

print("Minima Result: ", minima)
print("Iterations: ", n_iter)

minima = np.asarray(minima).reshape(-1)
fitted_freq = fitting_func(GW190412_time, *minima)
resid = GW190412_freq - fitted_freq

# estimate observational noise sigma from residuals
sigma_est = np.std(resid)
if sigma_est <= 0 or not np.isfinite(sigma_est):
    sigma_est = 1.0
print("Estimated sigma from residuals:", sigma_est)

# plt.scatter(GW190412_time, GW190412_freq)
# plt.plot(GW190412_time, fitting_func(GW190412_time, *minima))
# plt.show()
#def posterior(x, fit = fitting_func, data = (xpos,ypos)):
#    #s -> signal, just our model. 
#    log_likelihood = cost_func(fit, data, x) #gaussian distribution of noise
#    prior = 1 #uniform priors
#    return -1/2 * log_likelihood

#def proposal(x):
#    return np.random.uniform(-10,10) + x

#def markov(initial, post, prop, iterations):
#    x = [initial]
#    p = [post(x[-1])]
#    for i in range(iterations):
#        x_test = [prop(x[-1][i]) for i in range(7)]
#        p_test = post(x_test)
#        
#        acc = p_test/p[-1]
#        u = np.random.uniform(0,1)
#        if u <= acc:
#            x.append(np.array(x_test).ravel())
#            p.append(p_test)
#
#    return x, p
#
# def ensemble(nwalkers, initial, iterations):
#     chains = []
#     for i in range(nwalkers):
#         chain, prob = markov(initial, posterior, proposal, iterations)
#         chains.append(chain[-1])
#     return chains

