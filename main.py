import argparse
import optimize # your optimizer
import markov # my mcmc
from markov import rw_scales, strain_path, fs, center_time
import numpy as np
import os
import matplotlib.pyplot as plt
import corner #corner for corner plots
import emcee

# output dir
outdir = "mcmc_out"
os.makedirs(outdir, exist_ok=True)

# main loop
def main():
    parser = argparse.ArgumentParser(
        description="Fit to GW150914 using different MCMC parameters."
    )
    parser.add_argument(
        "--plot",
        choices=[
            "mass_scatter",
            "traces",
            "full_signal",
            "chirp",
            "corner",
            "emcee"
        ],
        required=True,
        help=(
            "Choose which plot to generate: "
            "'mass_scatter' for different walkers finding the values of m1 and m2, "
            "'traces' for walkers converging to a final mass value, "
            "'full_signal' for the original time domain LIGO waveform, "
            "'chirp' for the LIGO waveform processed to just the chirp, "
            "'corner' for corner plot of parameter posteriors, "
            "'emcee' for side-by-side of chains produced by emcee and homebrew markov "
        ),
    )
    parser.add_argument(
            "--nwalkers",
            type = int,
            default = 4,
            help=(
                "Number of walkers to use in ensemble sampling (defaults to 4)"
                )
            )
    parser.add_argument(
            "--nsteps",
            type = int,
            default = 1000,
            help=(
                "Number of iterations for Markov chains (defaults to 1000)"
                )
            )
    parser.add_argument(
            "--proposal_mode",
            choices=[
                "newton",
                "rw",
                "mix"
                ],
            default = "mix",
            help = (
                "Proposal mode to use for Markov chain (defaults to mix)"
                )
        )
    # loop from external library (emcee)
    #init = np.array([[35.0],
    #                 [30.0],
    #                 [400.0],
    #                 [center_time],
    #                 [0.0]]) #initialization, just GW150914 params
    #sampler = emcee.EnsembleSampler(nwalkers = nwalkers, ndim = init.shape[0], log_prob_fn = markov.log_posterior)
    #sampler.run_mcmc(init, nsteps)
    #chains = sampler.get_chain()
    
    args = parser.parse_args()

    nwalkers = args.nwalkers
    nsteps = args.nsteps
    proposal_mode = args.proposal_mode

    # main loop from markov.py
    chains, logs, accepts = markov.run_ensemble(nwalkers=nwalkers, nsteps=nsteps,
                                         proposal_mode=proposal_mode,
                                         rw_scales_local=rw_scales,
                                         adapt=True)
    print("Acceptance rates:", accepts)
    
    # removes first 30% of samples
    burnin = int(0.3 * nsteps)
    all_samples = np.vstack([chains[i][burnin:] for i in range(len(chains))])
    np.save(os.path.join(outdir, "posterior_samples.npy"), all_samples)
    print("Collected samples:", all_samples.shape)

    # printing final values
    labels = ["m1", "m2", "dL", "tc", "phi"]
    for i, lab in enumerate(labels):
        med = np.median(all_samples[:, i])
        lo = np.percentile(all_samples[:, i], 5)
        hi = np.percentile(all_samples[:, i], 95)
        print(f"{lab}: median={med:.5g}, 90% CI = [{lo:.5g}, {hi:.5g}]")
    

    chains_burnin = [chains[i][burnin:] for i in range(len(chains))]
    
    def gelman_rubin(chains, Ndims = 5):
        """
        This computes the Gelman-Rubin statistic for an ensemble of Markov chains by comparing the variance within a chain to the variance between chains. Note: this is done after burn-in values have been discarded.
        """
        chains_burnin = [chains[i][burnin:] for i in range(len(chains))] #chains (discarding burn-in)
        L = chains_burnin[0].shape[0] #length of each chain
        gelman_stats = np.zeros(Ndims) #array to store gelman stats

        chain_means, chain_var = [],[]
        for i in range(Ndims):
            for j in range(len(chains)):
                chain_means.append(np.mean(chains[j][:,i])) #take within chain mean of chain j, ith parameter
                chain_var.append(np.var(chains[j][:,i])) #within chain variance
            mean_chain_mean = np.mean(chain_means) #mean of chain means
            B = np.var(chain_means) #variance of chain means
            W = np.mean(chain_var) #mean of individual chain variances
            
            gelman_stat = (((L-1)/L * W + B/L))/W
            gelman_stats[i] += gelman_stat

        return gelman_stats

    gelman = gelman_rubin(chains)
    for i,c in enumerate(gelman):
        print(f"Gelman-Rubin statistic for parameter {labels[i]}: {c}")

    # plot traces of mass for convergence
    def mass_scatter():
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        for w in range(len(chains)):
            plt.plot(chains[w][:, 0], alpha=0.6)
        plt.ylabel("m1")
        plt.subplot(2, 1, 2)
        for w in range(len(chains)):
            plt.plot(chains[w][:, 1], alpha=0.6)
        plt.ylabel("m2")
        plt.xlabel("step")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "traces_m1m2.png"), dpi = 300)

    # plot mass posterior
    def mass_post():
        plt.figure(figsize=(5, 5))
        plt.scatter(all_samples[:, 0], all_samples[:, 1], s=2, alpha=0.4)
        plt.xlabel("m1")
        plt.ylabel("m2")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "m1_m2_scatter.png"), dpi = 300)
    
    #plot a sample chain for each parameter to visually confirm convergence
    def parameter_chains():
        fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (8.6))
        axs[0,0].plot(

    #plot time series strain data from detector (no whitening/filtering)
    def time_series():
        times, strain = markov.load_strain_text(strain_path, fs)
        plt.figure(figsize=(8,6))
        plt.plot(times, strain)
        plt.xlabel("Time (s)")
        plt.ylabel("Strain")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "time_series_GW150914.png"), dpi = 300)

    #plot corner plot of parameters
    def corner_plot():
        fig = corner.corner(all_samples, labels = labels, show_titles = True)
        plt.savefig(os.path.join(outdir, "posterior_corner.png"), dpi = 300)

    if args.plot == "corner":
        corner_plot()
    if args.plot == "full_signal":
        time_series()
main()
