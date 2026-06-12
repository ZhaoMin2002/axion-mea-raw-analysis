"""Functions to read raw binary data."""

import re
import os
from functools import partial

from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.io import loadmat


def read_raw(raw_path, well_shape=(6, 8), electrode_shape=(4, 4),
             start_end=None, channel_array_path=None):
    """Reads signal and metadata from .raw data.

    Parameters
    ----------
    raw_path : str
        Path to .raw file.
    well_shape : tuple of (float, float), optional, default: (6, 8)
        Shape of wells.
    electrode_shape : tuple of (float, float), optional, default: (4, 4)
        Shape of electrodes.
    start_end : tuple of (int, int), optional, default: None
        Start and end, in seconds, to slice the signal.
        Useful for limiting memory of large recordings.
    channel_array_path : str, optional, default: None
        Path to channel_array.mat used to infer the shape of the signal.
        If None, assumes this file is in the pwd. This is a struct created
        from the ChannelArray class provided by the Axis Biosystems.

    Returns
    -------
    sig : 5d array
        Array of (rows_well, cols_well, rows_elect, cols_elect, timepoints).

    Notes
    -----
    The returned signal corresponds to plate map.

    Well indices follow:

    [[A1, A2 ... A8],
     [B1, B2 ... B8],
     ...
     [F1, F2 ... F8]]

    Electrode indices follow:

    [[11, 21, 31, 41],
     [12, 22, 32, 42],
     [13, 23, 33, 43],
     [14, 24, 34, 44]]
    """

    # Read data
    with open(raw_path, 'rb') as f:
        f = f.read()

    # Split at raw string to remove pre meta data
    meta_raw, sig = f.split(b'Raw\x06\xac\x18\x9b')
    del f

    # Parse metadata
    fs = re.search('Sampling Frequency,.*?Hz', str(meta_raw)).group(0)
    fs = fs.split(',')

    scale = re.search('Scale,.*?V/sample', str(meta_raw)).group(0)
    scale = scale.split(',')

    meta = {}
    meta['fs'] = float(fs[1].split(' ')[0])
    meta['scale'] = float(scale[1].split(' ')[0])
    meta['units'] = {
        'fs': fs[1].split(' ')[1],
        'scale': scale[1].split(' ')[1]
    }

    # Unpack dimensions
    n_rows_well, n_cols_well = well_shape
    n_wells = int(n_rows_well * n_cols_well)

    n_rows_electrodes, n_cols_electrodes = electrode_shape
    n_electrodes = int(n_rows_electrodes * n_cols_electrodes)

    # Get well labels (WT/WT vs WT/delE9)
    end_pos = 1000000 - re.search(b'.{17}\x81h\xackQ\?M', sig[-1000000:]).start()
    match = r'\\x0.\\x00\\x00\\x01\\x.[^3]\\x..\\x..\\x0.\\x00\\x00\\x00.*?x0'
    match = re.findall(match, str(sig[-end_pos:]))

    meta['well_labels'] = np.array([i[44:-3] for i in match]).reshape(n_rows_well, n_cols_well)

    # Slice
    if start_end is not None:
        # Trim to start_end
        start, end = start_end

        start = int(start * meta['fs'] * 1000)
        end = int(end * meta['fs'] * 1000)

        # 2 bytes per int16
        start = int(2 * start * n_wells * n_electrodes)
        end   = int(2 * end   * n_wells * n_electrodes)

        sig = np.frombuffer(sig[:-end_pos][start:end], 'int16')

    else:
        # Remove metadata bytes from EOF
        sig = np.frombuffer(sig[:-end_pos], 'int16')

    # Reshape signal - at this point the indices are
    #   of the wells and electrodes are still scambled
    shape_rev = (n_cols_electrodes, n_rows_electrodes,
                 n_cols_well, n_rows_well)

    sig = sig.reshape(-1, *shape_rev).T

    # Channel array specification generated in matlab.
    #   Corresponds to the MEA config: 48WellTransparent
    if channel_array_path is None:
        channel_array_path = '.'

    if not channel_array_path.endswith('.mat'):
        channel_array_path = f"{channel_array_path}/channel_array.mat"

    channel_array = loadmat(channel_array_path)
    row_names = list(channel_array['out'].dtype.names)

    # Unpack .mat struct
    channel_dict = {}
    for name, values in zip(row_names, channel_array['out'][0][0]):
        channel_dict[name] = values[0]

    # Get indices of wells and electrodes
    wrow_inds = channel_dict['WellRow'].reshape(*shape_rev).T
    wcol_inds = channel_dict['WellColumn'].reshape(*shape_rev).T

    erow_inds = channel_dict['ElectrodeRow'].reshape(*shape_rev).T
    ecol_inds = channel_dict['ElectrodeColumn'].reshape(*shape_rev).T

    # Reshape signal into logical structure
    locs = np.zeros((*shape_rev[::-1], 4), dtype=int)
    for wr in range(n_rows_well):
        for wc in range(n_cols_well):
            for er in range(n_rows_electrodes):
                for ec in range(n_cols_electrodes):

                    loc = np.where(
                        (wrow_inds == wr+1) &
                        (wcol_inds == wc+1) &
                        (erow_inds == er+1) &
                        (ecol_inds == ec+1)
                    )

                    locs[wr, wc, er, ec] = np.array(loc).T[0]

    locs = locs.reshape(-1, 4).T

    sig = sig[locs[0], locs[1], locs[2], locs[3], :]
    sig = sig.reshape(*shape_rev[::-1], -1)

    return sig, meta


def window_signal(sig, fs, output_dir, win_len=1.,
                  compressed=True, n_jobs=1, progress=None):
    """Split and save signal into smaller windows.

    Parameters
    ----------
    sig : 5d array
        Array of (rows_well, cols_well, rows_elect, cols_elect, timepoints).
    fs : float
        Sampling rate, in Hertz.
    output_dir : str
        Where to save results to.
    win_len : float, optional, default: 1
        Length per window, in seconds.
    compressed : bool, optional, default: True
        Save files as compressed .npz when True.
    n_jobs : int, optional, default: 1
        Number of jobs to save in parallel.
        -1 default to all available cpus.
    progress : {tqdm.tqdm, tqdm.notebook.tqdm}
        Use progress bar,
    """

    # Ensure directory exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Window indices
    n_samples = int(win_len * fs)
    n_segments = int(np.floor(sig.shape[-1] / n_samples))

    # Save
    if n_jobs == 1:

        inds = (np.arange(n_segments) * n_samples).astype(int)
        starts = inds[:-1]
        ends = inds[1:]

        if progress is not None:
            iterable = progress(range(len(starts)), total=len(starts))
        else:
            iterable = range(len(starts))

        for ind in iterable:
             _save_signal((sig[:, :, :, :, starts[ind]:ends[ind]], out[ind]),
                          compressed=compressed)

    else:

        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        # Output file names
        out = [f'{output_dir}/sig_{str(ind).zfill(4)}.npz'
               for ind in range(n_segments)]

        # Reshape signal
        sig = sig[:, :, :, :, :int(n_segments * fs)].reshape(*sig.shape[:-1], -1)
        sig = sig.reshape(*sig.shape[:-1], n_segments, -1)
        sig = np.moveaxis(sig, 4, 0)

        with Pool(processes=n_jobs) as pool:

            mapping = pool.imap(partial(_save_signal, compressed=compressed),
                                zip(sig, out))

            if progress is None:
                list(mapping)
            else:
                list(progress(mapping, total=len(out)))

    del sig


def load_windows(dir_path, inds):
    """Load windows from .npz files.

    Parameters
    ----------
    dir_path : str
        Path to where .npz are saved.
    inds : int or list of int
        Window indices to return.

    Returns
    -------
    sig : 5d or 6d array
        Either a signal as:
        (rows_well, cols_well, rows_elect, cols_elect, timepoints)
        when an int is is passed to inds.

        Or as:
        (n_windows, rows_well, cols_well, rows_elect, cols_elect, timepoints)
        when a list of int is passed to inds.

    Notes
    -----
    Assumes files in dir_path are save from window_signal.
    """
    if isinstance(inds, int):
        ind = str(inds).zfill(4)
        sig = np.load(f'{dir_path}/sig_{ind}.npz')['arr_0']
    else:
        for i, ind in enumerate(inds):
            _sig = load_windows(dir_path, ind)

            if i == 0:
                sig = np.zeros((len(inds), *_sig.shape))

            sig[i] = _sig

    return sig


def _save_signal(sig, compressed=True):
    """Save array."""

    # Unpack
    sig, out = sig

    # Compressed (or not)
    fsave = np.savez_compressed if compressed else np.savez

    # Save
    fsave(out, sig)
