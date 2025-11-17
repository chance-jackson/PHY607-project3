import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, LinAlgError
import optimize          # your optimize.py
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
nsteps = 8000
proposal_mode = "newton"       # "rw", "newton", or "mix"
# proposal stds for [m1, m2, dL, tc, phi]
rw_scales = np.array([1.5, 1.5, 50.0, 0.0008, 0.5])

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
