import argparse
import optimize # your optimizer
import markov # my mcmc
from markov import nwalkers, nsteps, proposal_mode, rw_scales
import numpy as np
import os
import matplotlib.pyplot as plt

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
            "chirp"
        ],
        required=True,
        help=(
            "Choose which plot to generate: "
            "'mass_scatter' for different walkers finding the values of m1 and m2, "
            "'traces' for walkers converging to a final mass value, "
            "'full_signal' for the original time domain LIGO waveform, "
            "'chirp' for the LIGO waveform processed to just the chirp "
        ),
    )





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
    plt.savefig(os.path.join(outdir, "traces_m1m2.png"))

    # plot mass posterior
    plt.figure(figsize=(5, 5))
    plt.scatter(all_samples[:, 0], all_samples[:, 1], s=2, alpha=0.4)
    plt.xlabel("m1")
    plt.ylabel("m2")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "m1_m2_scatter.png"))

    print("Saved outputs in", outdir)
