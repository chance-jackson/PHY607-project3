import argparse
import optimize # your optimizer
import markov # my mcmc
from markov import proposal_mode, rw_scales, strain_path, fs, center_time
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
    
    def time_series():
        times, strain = markov.load_strain_text(strain_path, fs)
        plt.figure(figsize=(8,6))
        plt.plot(times, strain)
        plt.xlabel("Time (s)")
        plt.ylabel("Strain")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "time_series_GW150914.png"), dpi = 300)

    def corner_plot():
        fig = corner.corner(all_samples, labels = labels, show_titles = True)
        plt.savefig(os.path.join(outdir, "posterior_corner.png"), dpi = 300)
    #print("Saved outputs in", outdir)

    if args.plot == "corner":
        corner_plot()
    if args.plot == "full_signal":
        time_series()
main()
