"""Compute spectral across high-dimensional MEA array."""

import os

import numpy as np
from scipy.signal import resample

from neurodsp.spectral import compute_spectrum
from timescales.autoreg import compute_ar_spectrum

from .io import load_windows



def compute_spectra_windows(sig_dir, fs, nperwindow, n_resample=None, ar_order=None,
                            apply_funcs=None, out_dir=None, compressed=False, progress=None,
                            **kwargs):
    """Compute windowed spectra.

    Parameters
    ----------
    sig_dir : str
        Path to .npz files, save from window_signal.
    fs : float
        Sampling rate, in Hertz.
    nperwindow : int
        Number of files to combine per window.
    n_resample : int, optional, default: None
        Number of samples to resample to. Redefines sampling rate.
        Downsampling is particularly important when using AR models
        with a low high frequency cutoff, relative to the Nyquist frequency.
    ar_order : int, optional, default: None
        Autoregressive order. Welch's is used if None.
    apply_funcs : [list of] tuple of ({'all', 'well'}, func), optional, default: None
        Applies a preprocessing function across signals.
        If the string is 'all', the function is applied to each signal, and should
        accept a 1d array and return a 1d array.
        If the string is 'wells', the function is applied to each well, and should
        accept a 2d array and return a 2d or 1d array.
        If a list of tuples is passed, functions are applied in order.
    out_dir : str, optional, default: None
        Save out spectra for future use.
    compressed : bool, optional, default: False
        Save PSD as a compressed .npz file.
        Saving and loading can be very slow if compressed.
    progress : {'tqdm.tqdm', 'tqdm.notebook.tqdm'}
        Progress bar for the impatient.
    **kwargs
        Additional kwargs to pass to compute_spectrum
        or compute_ar_spectrum.

    Returns
    -------
    freqs : 1d array
        Frequency definition.
    powers : 5d or 6d array
        Powers as:
        ([n_windows,] n_well_rows, n_well_cols,
         n_electrode_rows, n_electrode_cols, n_powers)
    """

    n_files = len(os.listdir(sig_dir))
    n_windows = int(np.floor(n_files / nperwindow))

    if n_files == 0:
        raise ValueError("No valid signals in sig_dir.")

    # Define iterator
    if progress is not None:
        f_iter = progress(enumerate(range(0, n_files-nperwindow, nperwindow)),
                          total=n_windows)
    else:
        f_iter = enumerate(range(0, n_files-nperwindow, nperwindow))

    # Iterate over windows
    for pind, ind in f_iter:

        # Load signal
        sig = load_windows(sig_dir, list(range(ind, ind+nperwindow)))
             
        if n_resample is not None and pind == 0:
            # Redefine sampling rate
            fs = n_resample / (sig.shape[-1] / fs)

        if n_resample is not None:
            # Resample
            sig = resample(sig, n_resample, axis=-1)

        if nperwindow > 1:
            # Move n_windows axis to -2 position
            sig = np.moveaxis(sig, 0, -2)

            # Flatten voltages across nwindows
            sig = sig.reshape(*sig.shape[:-2], int(sig.shape[-2] * sig.shape[-1]))

        # Preprocessing
        if apply_funcs is not None:

            # Make iterable
            if isinstance(apply_funcs[0], str):
                apply_funcs = [apply_funcs]

            # Get shape
            ncols, nrows, ncols_e, nrows_e = sig.shape[:-1]

            # Step through preprocessing functions
            for mode, afunc in apply_funcs:
                _sig = None
                # Iterate over wells
                for wc in range(ncols):
                    for wr in range(nrows):

                        if mode == 'all':
                            # Apply step to each signal
                            for ec in range(ncols_e):
                                for er in range(nrows_e):
                                    sig[wc, wr, ec, er] = afunc(sig[wc, wr, ec, er])
                        else:
                            # Apply step on a well-by-well basis
                            sig_mod = afunc(sig[wc, wr])

                            # Infer returned shape
                            if sig_mod.ndim == 3:
                                # Maintains electrode dims
                                sig[wc, wr] = sig_mod
                            elif sig_mod.ndim == 1 and wc == 0 and wr == 0:
                                # Collapse electrode dims
                                _sig = np.zeros((ncols, nrows, sig.shape[-1]))
                                _sig[wc, wr] = sig_mod
                            else:
                                _sig[wc, wr] = sig_mod

                if _sig is not None:
                    sig = _sig

        # Track original shape for later to reshape to/from 2d
        orig_shape = sig.shape

        # Reshape to 2d - required for computing spectra
        sig = sig.reshape(-1, orig_shape[-1])

        # Compute spectra
        if ar_order is None:
            freqs, _powers = compute_spectrum(sig, fs, **kwargs)
        else:
            freqs, _powers = compute_ar_spectrum(sig, fs, ar_order, **kwargs)

        # Reshape powers back into well and electrode shape
        _powers = _powers.reshape(*orig_shape[:-1], -1)

        # Initialize full array
        if ind == 0:
            powers = np.zeros((n_windows, *_powers.shape))

        powers[pind] = _powers

    if out_dir is not None:
        # Save to output path
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        save = np.savez_compressed if compressed else np.savez
        save(f'{out_dir}/freqs', freqs)
        save(f'{out_dir}/powers', powers)

    return freqs, powers
