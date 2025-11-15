import numpy as np
import matplotlib.pyplot as plt
import optimize
import emcee
import corner

########### Import grav wave data ############


def fitting_func(x,a,b,c,d,e,f,g):
    return g * x ** 6 + f * x ** 5 + e * x ** 4 + a * x ** 3 + b * x ** 2 + c * x + d

def cost_func(fit, data, guess_params):
    resid = fit(data[0], *guess_params) - data[1] #find resid
    square_resid = np.power(resid,2) #square em

    return  np.sum(square_resid) #return their sum

def posterior(x, fit = fitting_func, data = (xpos,ypos)):
    #s -> signal, just our model. 
    log_likelihood = cost_func(fit, data, x) #gaussian distribution of noise
    prior = 1 #uniform priors
    return -1/2 * log_likelihood

def proposal(x):
    return np.random.uniform(-10,10) + x

def markov(initial, post, prop, iterations):
    x = [initial]
    p = [post(x[-1])]
    for i in range(iterations):
        x_test = [prop(x[-1][i]) for i in range(7)]
        p_test = post(x_test)
        
        acc = p_test/p[-1]
        u = np.random.uniform(0,1)
        if u <= acc:
            x.append(np.array(x_test).ravel())
            p.append(p_test)

    return x, p

initial_random = np.zeros(7)
for i in range(7):
    initial_random[i] += np.random.uniform(-1000,1000)
print(initial_random)
def ensemble(nwalkers, initial, iterations):
    chains = []
    for i in range(nwalkers):
        chain, prob = markov(initial, posterior, proposal, iterations)
        chains.append(chain[-1])
    return chains

minima, x_hist, y_hist, n_iter = optimize.newtons(lambda guess_params: cost_func(fitting_func, (xpos, ypos), guess_params), np.ones(7), rate = 1)

#print(markov(minima, posterior, proposal, 1000)[-1])
#walker_steps = [chain[i][0] for i in range(len(chain))]
#plt.plot(walker_steps)
#plt.show()
#sampler = emcee.EnsembleSampler(500, 7, posterior, args = (xpos,ypos))
#N_steps = 1000
#sampler.run_mcmc(minima, N_steps)
#flat_samples = sampler.get_chain(discard = 100, thin = 15, flat = True)
labels = ["a", "b", "c", "d", "e", "f", "g"]
#fig = corner.corner(flat_samples, labels = labels)
#plt.show()
pos = ensemble(10, minima, 10)
#print(np.array(pos))
chain, prob = markov(initial_random, posterior, proposal, 100000)
#print(chain[0][0])
param_1 = np.zeros(len(chain))
param_2 = np.zeros(len(chain))
param_3 = np.zeros(len(chain))
param_4 = np.zeros(len(chain))
param_5 = np.zeros(len(chain))
param_6 = np.zeros(len(chain))
param_7 = np.zeros(len(chain))
for i in range(len(chain)):
    param_1[i] += chain[i][0]
    param_2[i] += chain[i][1]
    param_3[i] += chain[i][2]
    param_4[i] += chain[i][3]
    param_5[i] += chain[i][4]
    param_6[i] += chain[i][5]
    param_7[i] += chain[i][6]
rand_idx = np.random.randint(0,len(chain))
print(rand_idx)
rand_params = (param_1[rand_idx],param_2[rand_idx],param_3[rand_idx],param_4[rand_idx], param_5[rand_idx], param_6[rand_idx], param_7[rand_idx])
plt.clf()

plt.scatter(xpos, ypos)
plt.scatter(xpos,fitting_func(xpos, *rand_params),color='red')
plt.show()
#print(param_1)
#print(np.array(pos).flatten())
#need one single parameter chain from pos chain:
#param_1 = [pos[
tau = emcee.autocorr.integrated_time(param_1, quiet = True)[0]
print(chain[-1])
print(minima)
fig, axs = plt.subplots(ncols = 4, nrows = 2, figsize=(10,4),layout="constrained",sharex=True) 
axs[0,0].plot(param_1,label="Parameter 1")
axs[0,1].plot(param_2, label = "Parameter 2")
axs[0,2].plot(param_3, label = "Parameter 3")
axs[0,3].plot(param_4, label = "Parameter 4")
axs[1,0].plot(param_5, label = "Parameter 5")
axs[1,1].plot(param_6, label = "Parameter 6")
axs[1,2].plot(param_7, label = "Parameter 7")

axs[0,0].set_title("Parameter 1")
axs[0,1].set_title("Parameter 2")
axs[0,2].set_title("Parameter 3")
axs[0,3].set_title("Parameter 4")
axs[1,0].set_title("Parameter 5")
axs[1,1].set_title("Parameter 6")
axs[1,2].set_title("Parameter 7")

fig.supxlabel("Iteration")
fig.supylabel("Parameter Value")
plt.show()

fig = corner.corner(np.array(pos), labels = labels, quantiles = [0.16, 0.5, 0.84], show_titles = True)
fig.suptitle("Corner plot After 1000 Walkers, 1000 iterations")
plt.show()
#print(pos)
