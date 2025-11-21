import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, LinAlgError
import optimize          # your optimize.py, we end up needing to use multiple optimizers due to inaccuracies but we can discuss in report
from scipy.signal import welch
from pycbc.waveform import get_fd_waveform
from pycbc.psd import inverse_spectrum_truncation
from pycbc.types import FrequencySeries
from math import pi

# output dir
outdir = "mcmc_out"
os.makedirs(outdir, exist_ok=True)

# modeling GW150914
## import livingston data
strain_path = "L-L1_GWOSC_16KHZ_R1-1126259447-32.txt"

fs = 16384.0 # sampling rate (Hz) of the strain file
f_low = 20.0 # low-frequency cutoff for analysis (Hz)
delta_t_window = 8.0 # seconds of data window around event to analyze
approximant = "IMRPhenomD" # picking inspiral–merger–ringdown model

# GW150914 event time (for picking window only; tc is still a free parameter)
file_start_gps = 1126259446.0      # start GPS of your file, strain file doesnt come with relative time
event_gps      = 1126259462.4      # approximate merger time
event_rel_time = event_gps - file_start_gps 

# MCMC settings
nwalkers = 4
nsteps = 1000
proposal_mode = "mix"       # "rw", "newton", or "mix"
# proposal stds for [m1, m2, dL, tc, phi]
rw_scales = np.array([3.0, 3.0, 150.0, 0.002, 1.0])


def load_strain_text(path, fs):
    """ Load 1-column strain file and build time array. """
    strain = np.loadtxt(path)
    times = np.arange(len(strain)) / fs
    return times, strain

def freqs_and_spec_from_timeseries(strain_ts, fs, nfft=None):
    """ Converts time-domain strain data into frequency domain data. """
    if nfft is None:
        nfft = int(2**np.ceil(np.log2(len(strain_ts)))) # length of FFT
    spec = np.fft.rfft(strain_ts, n=nfft) # compute real FFT
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs) # build frequency array
    return freqs, spec

def inner_product(a, b, psd_interp, df):
    """ Inner product used in LIGO likelihood function, noise weighted. """
    return 4.0 * df * np.real(np.sum(a * np.conjugate(b) / psd_interp))

def complex_interp(x, xp, yp):
    """ Interpolate complex waveform onto frequency array. """
    return (np.interp(x, xp, yp.real) +
            1j * np.interp(x, xp, yp.imag))

# Load data
if not os.path.exists(strain_path):
    raise FileNotFoundError(f"No file found: {strain_path}")

times, strain = load_strain_text(strain_path, fs)

# subtract DC to avoid huge low-f noise
strain = strain - np.mean(strain)

# pick an 8 s window around the known event time, since files are large
center_time = event_rel_time
start_time = center_time - delta_t_window / 2
end_time   = center_time + delta_t_window / 2

mask = (times >= start_time) & (times < end_time)
tseg = times[mask]
hseg = strain[mask]

# FFT of the data segment
nfft = int(2**np.ceil(np.log2(len(hseg))))
data_freqs, data_spec = freqs_and_spec_from_timeseries(hseg, fs, nfft=nfft)
df = data_freqs[1] - data_freqs[0]

# noise PSD estimation, to understand what frequencies are noisey/sensitive
off_mask = ~mask 

# this uses welch's method to compute PSD of only noise section
f_w, psd_w = welch(strain[off_mask], fs=fs, nperseg=int(2 * fs))

psd_w[psd_w <= 0] = 1e-50  # avoid zeros / negatives

# interpolate PSD to data frequency grid
psd_interp = np.interp(data_freqs, f_w, psd_w)

# wrap in FrequencySeries(PyCBC func) and apply inverse spectrum truncation(smoothing PSD to avoid fitting to noise)
psd_fs = FrequencySeries(psd_interp, delta_f=df)
trunc_len = min(len(psd_fs), int(4 * fs / df))
psd_fs = inverse_spectrum_truncation(psd_fs, trunc_len, low_frequency_cutoff=f_low)

# back from PyCBC object to numpy array
psd_interp = np.array(psd_fs)

# restrict to f >= f_low
valid = data_freqs >= f_low
data_freqs = data_freqs[valid]
data_spec  = data_spec[valid]
psd_interp = psd_interp[valid]

def make_waveform_fd(m1, m2, dL, phi_c, delta_f, f_lower=f_low, approximant=approximant):
    """
    Generate model waveform with PyCBC.
    Inputs:
        m1, m2: solar masses
        dL: luminosity distance
        phi_c: coalescence phase
        delta_f: spacing between bins
        f_lower: minimum frequency to use
        approximant: waveform model

    Ouputs:
        hp: array of + polarization
        hp.sample_frequencies: frequency grif
    """
    try:
        hp, hc = get_fd_waveform(approximant=approximant,
                                 mass1=float(m1), mass2=float(m2),
                                 delta_f=delta_f, f_lower=f_lower,
                                 distance=float(dL), phase=float(phi_c))
    except Exception:
        return None, None
    return np.array(hp), np.array(hp.sample_frequencies)

def align_waveform_to_data(hp_arr, hp_freqs, data_freqs, tc=0.0):
    """
    Matching PyCBC frequency grif to data grid from FFT. 
    """
    if data_freqs[0] < hp_freqs[0]:
        return None

    hp_interp = complex_interp(data_freqs, hp_freqs, hp_arr).astype(np.complex128)
    phasor = np.exp(-2j * np.pi * data_freqs * tc)
    return hp_interp * phasor

def log_prior(params):
    m1, m2, dL, tc, phi = params

    # mass bounds for assumed values
    if not (20.0 < m2 < m1 < 60.0):
        return -np.inf

    # we know this is symmetrical so assume we start there
    q = m2 / m1
    if not (0.6 < q < 1.0):
        return -np.inf

    # distance bounds
    if not (200.0 < dL < 800.0):  # Mpc
        return -np.inf

    # restricts merger to 8 second window
    if not (tseg[0] - 0.1 < tc < tseg[-1] + 0.5):
        return -np.inf

    # phase
    if not (0.0 <= phi < 2 * np.pi):
        return -np.inf

    # source uniformly distributed in volume
    return 2.0 * np.log(dL)

def log_likelihood(params):
    m1, m2, dL, tc, phi = params

    # generate model
    hp_arr, hp_freqs = make_waveform_fd(m1, m2, dL, phi,
                                        delta_f=df, f_lower=f_low,
                                        approximant=approximant)
    if hp_arr is None:
        return -np.inf

    # align data to same freq grid
    model = align_waveform_to_data(hp_arr, hp_freqs, data_freqs, tc=tc)
    if model is None:
        return -np.inf

    # compute residuals between real FFT and model
    resid = data_spec - model
    chi2 = inner_product(resid, resid, psd_interp, df)
    # gaussian likelihood
    return -0.5 * chi2

def log_posterior(params):
    lp = log_prior(params)
    # if prior rejects
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params)
    return lp + ll

# cost wrapper
def cost_for_optimize(x):
    # convert input to array
    x = np.asarray(x).ravel()
    if x.size == 2:
        m_chirp, tc = float(x[0]), float(x[1])
        m_equal = m_chirp * (2.0 ** (1.0 / 5.0))  # Mc = m * 2^-1/5, for equal mass
        # 5 paramter guess
        params = [m_equal, m_equal, 400.0, tc, 0.0]
    elif x.size == 5:
        # optimizer already predicts 5
        params = x
    else:
        raise ValueError("cost_for_optimize expects 2 or 5 params")

    lp = log_posterior(params)
    # prior rejects
    if not np.isfinite(lp):
        return 1e30
    return -lp


# Proposals
rng = np.random.default_rng()

def proposal_rw(theta, scales):
    return theta + rng.normal(scale=scales)

def proposal_newton(
        theta, f_obj, 
        delta_x=1e-3, 
        alpha=0.03, 
        cov_scale=0.005):
    """ Newton-informed Gaussian proposal using optimize. """
    try:
        g = optimize.grad(f_obj, theta, delta_x).ravel()
        H = optimize.hessian(f_obj, theta, delta_x)
    except Exception:
        return None, False

    H = 0.5 * (H + H.T) # symmetric hessian
    eps = 1e-6
    try:
        Hreg = H + eps * np.eye(H.shape[0]) # making the matrix invertible by adding eps
        Hinv = inv(Hreg) # inverts
    except (LinAlgError, ValueError):
        return None, False

    center = theta - alpha * (Hinv @ g) # proposal mean
    cov = cov_scale * Hinv 
    cov = 0.5 * (cov + cov.T) + 1e-8 * np.eye(cov.shape[0]) # covariance matrix
    try:
        prop = rng.multivariate_normal(mean=center, cov=cov) # random proposal
    except Exception:
        return None, False
    return prop, True

# MCMC sampler
def markov_mh(initial, nsteps=2000, proposal="rw",
              rw_scales_local=None, adapt=False, adapt_interval=1000):
    
    # initialize paramters in chain and posterior values
    theta = np.array(initial, dtype=float).ravel()
    ndim = len(theta)
    chain = np.zeros((nsteps, ndim))
    logpost_chain = np.full(nsteps, -np.inf)
    accepts = 0

    # random walk step sizing
    if rw_scales_local is None:
        rw_scales_local = rw_scales.copy()

    # current position
    cur_lp = log_posterior(theta)

    # main loop
    for i in range(nsteps):
        # convert this to options in code ? later
        if proposal == "rw":
            prop = proposal_rw(theta, rw_scales_local)
        # in general we will use the optimizer
        elif proposal == "newton":
            prop, used = proposal_newton(theta, cost_for_optimize,
                                         delta_x=1e-3, alpha=0.03, cov_scale=0.005)
            if prop is None:
                prop = proposal_rw(theta, rw_scales_local)
        elif proposal == "mix":
            if rng.random() < 0.9:
                prop = proposal_rw(theta, rw_scales_local)
                used = False
            else:
                prop, used = proposal_newton(theta, cost_for_optimize,
                                            delta_x=1e-3, alpha=0.03, cov_scale=0.005)
                if prop is None:
                    prop = proposal_rw(theta, rw_scales_local)
        else:
            raise ValueError("Unknown proposal mode")

        # posterior of proposal
        prop_lp = log_posterior(prop)
        # acceptance condition
        if not np.isfinite(prop_lp):
            chain[i] = theta
            logpost_chain[i] = cur_lp
        else:
            accept_prob = min(1.0, np.exp(prop_lp - cur_lp))
            if rng.random() < accept_prob:
                theta = prop
                cur_lp = prop_lp
                accepts += 1
            chain[i] = theta
            logpost_chain[i] = cur_lp

        # adaptive tuning on RW scales
        if adapt and (i + 1) % adapt_interval == 0:
            recent_accept = accepts / float(i + 1)
            target = 0.25
            # acceptance too low
            if recent_accept < target * 0.7:
                rw_scales_local = rw_scales_local * 0.8
            # acceptance too high
            elif recent_accept > target * 1.3:
                rw_scales_local = rw_scales_local * 1.2
            print(f"[adapt] step {i+1}: acc~{recent_accept:.3f}, new scales {rw_scales_local}")

    acc_rate = accepts / float(nsteps)
    return chain, logpost_chain, acc_rate

# running the simulation
def run_ensemble(nwalkers=nwalkers, nsteps=nsteps,
                 proposal_mode=proposal_mode, rw_scales_local=rw_scales,
                 adapt=False):
    chains = []
    logs = []
    accepts = []

    # use optimizer to pick starting values
    try:
        # for GW150914, start near Mc~28 Msun
        seed = np.array([28.0, center_time])
        minima, _, _, nit = optimize.newtons(lambda x: cost_for_optimize(x),
                                             seed, delta_x=1e-3, rate=0.5, tol=1e-5)
        minima = np.asarray(minima).ravel()
        print("Newton seed minima (Mc, tc):", minima, "it:", nit)
        m_equal = minima[0] * (2.0 ** (1.0 / 5.0))
        base_init = np.array([m_equal, m_equal, 400.0, minima[1], 0.0])
    except Exception as e:
        print("Newton seeding failed:", e)
        # fallback guess given GW150914 params
        base_init = np.array([35.0, 30.0, 400.0, center_time, 0.0])

    for w in range(nwalkers):
        # walker initilazation
        jitter = np.array([
            0.05 * base_init[0] * np.random.randn(),
            0.05 * base_init[1] * np.random.randn(),
            50.0 * np.random.randn(),
            0.0005 * np.random.randn(),
            0.5 * np.random.randn()
        ])
        init = base_init + jitter
        print(f"Starting walker {w}, init = {init}")
        # run walkers
        ch, lp, acc = markov_mh(init, nsteps=nsteps, proposal=proposal_mode,
                                rw_scales_local=rw_scales_local, adapt=adapt)
        chains.append(ch)
        logs.append(lp)
        accepts.append(acc)
        # np.save(os.path.join(outdir, f"chain_w{w}.npy"), ch)
        # np.save(os.path.join(outdir, f"logpost_w{w}.npy"), lp)
        print(f"Walker {w} done. acc={acc:.3f}")

    return chains, logs, accepts


# main loop, move to seperate file later, we need to add more options maybe different files
if __name__ == "__main__":
    chains, logs, accepts = run_ensemble(nwalkers=nwalkers, nsteps=nsteps,
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
