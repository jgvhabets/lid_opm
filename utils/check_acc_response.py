import time
import numpy as np
from pylsl import StreamInlet, resolve_streams



ACC_LSL_IDX = {'left': [9, 10, 11], 'right': [6, 7, 8]}  # example channel indices for left and right ACC; adjust as needed


def create_acc_inlet(stream_name: str = 'acc', timeout: float = 5.0,
                     max_buffer_sec=25,verbose:bool=False) -> StreamInlet:
    """Resolve an ACC LSL stream and return a connected StreamInlet.

    Call this once before starting the task loop and pass the returned
    inlet to run_trial() via the acc_inlet parameter.
    """
    streams = resolve_streams()
    if not streams:
        raise RuntimeError(
            f"No LSL stream named '{stream_name}' found within {timeout} s. "
            "Make sure the ACC stream is broadcasting before starting the task."
        )
    
# print available streams only when verbose is True
    if verbose:
        print("Available LSL streams:")

    aux_stream = None

    for s in streams:
        if 'LID_MEG' in s.name() and not 'TRG' in s.name():
            aux_stream = s
            break
    if aux_stream is None:
        available_names = ", ".join(s.name() for s in streams)
        raise RuntimeError(
            f"No ACC stream found. Available streams: {available_names or 'None'}."
            "Make sure the ACC stream is broadcasting before starting the task."
        )
    if verbose:
        print(f'selected stream for ACC: {aux_stream.name()}')
        print(f'sampling rate: {aux_stream.nominal_srate()} Hz')
    
    aux_stream = StreamInlet(
        aux_stream,
        max_buflen=max_buffer_sec,  # keep at least this many SECONDS of data in the inlet buffer
        max_chunklen=int(aux_stream.nominal_srate() * max_buffer_sec)  # adjust max_chunklen as needed for expected sample rates and processing speed
    )

    # Print basic channel count
    if verbose:
        print(f"Number of channels: {aux_stream.info().channel_count()}")
        print(f"Sampling rate: {aux_stream.info().nominal_srate()}")

    # get baseline for acc hands
    base_left, base_right = get_acc_baselines(aux_stream)
    if verbose:
        print(f"ACC baseline - Left hand: {base_left:.2f}, Right hand: {base_right:.2f}")
    acc_bases = {'left': base_left, 'right': base_right}

    return aux_stream, acc_bases



def get_acc_baselines(inlet: StreamInlet, duration_sec: float = 10.0, verbose = False):

    ch_left = ACC_LSL_IDX['left']
    ch_right = ACC_LSL_IDX['right']

    if verbose:
        print(f"Collecting {duration_sec} seconds of baseline ACC data for left and right hands...")
    start_time = time.time()

    left_samples = []
    right_samples = []


    while time.time() - start_time < duration_sec:
        samples, timestamps = inlet.pull_chunk(timeout=0.1, max_samples=250)
        if samples:
            samples_left = np.array(samples)[:, ch_left]
            samples_right = np.array(samples)[:, ch_right]
            # detrend by removing mean of raw values (pos and neg)
            samples_left = samples_left - np.mean(samples_left)
            samples_right = samples_right - np.mean(samples_right)
            # compute RMS amplitude (vector length) for each channel group
            vector_left = np.sqrt(np.mean(samples_left ** 2))
            vector_right = np.sqrt(np.mean(samples_right ** 2))
            left_samples.append(vector_left)
            right_samples.append(vector_right)



    # plt.figure()
    # plt.plot(left_samples, label='Left hand ACC vectors')
    # plt.plot(right_samples, label='Right hand ACC vectors')
    # plt.title('ACC Baseline Samples')
    # plt.xlabel('Sample')
    # plt.ylabel('RMS Amplitude')
    # plt.legend()
    # plt.show()

    if not left_samples or not right_samples:
        if verbose:
            print("No ACC samples collected during baseline period; using fallback baseline thresholds.")
        return 1.0, 1.0

    # Compute baseline means
    baseline_left = np.mean(left_samples) + (np.std(left_samples) * 4)
    baseline_right = np.mean(right_samples) + (np.std(right_samples) * 4)

    return baseline_left, baseline_right


def compute_acc_euclidean_norm(acc_samples: np.ndarray, stim_direction: str) -> float:
    """Compute the Euclidean norm for the 3-axis ACC window on one side.

    The helper keeps the task logic aligned with the project style by
    operating on the raw buffered window and returning a single movement
    summary value for the selected side.
    """
    samples = np.asarray(acc_samples)
    if samples.size == 0:
        return float("nan")

    side_samples = samples[:, ACC_LSL_IDX[stim_direction]]
    side_samples = side_samples - np.mean(side_samples, axis=0)
    euclidean_norm = np.sqrt(np.sum(side_samples ** 2, axis=1))

    return float(np.max(euclidean_norm))


def calibrate_first_trial_threshold(
    inlet: StreamInlet,
    stim_direction: str,
    calibration_ms: float = 200.0,
    max_samples: int = 250,
    verbose: bool = False
) -> float:
    """Calibrate the movement threshold from a short fixation baseline.

    The helper collects ACC samples for the requested window, computes the
    Euclidean norm for each sample on the stimulated side, and returns
    mean + 3 * SD as the movement threshold.
    """
    if calibration_ms <= 0:
        return float("nan")

    start_time = time.time()
    baseline_samples = []

    while (time.time() - start_time) * 1000.0 < calibration_ms:
        samples, _timestamps = inlet.pull_chunk(timeout=0.01, max_samples=max_samples)
        if samples:
            side_samples = np.asarray(samples)[:, ACC_LSL_IDX[stim_direction]]
            baseline_samples.append(side_samples)

    if not baseline_samples:
        return float("nan")

    baseline_window = np.vstack(baseline_samples)
    euclidean_norms = np.sqrt(np.sum(baseline_window ** 2, axis=1))
    mean_norm = float(np.mean(euclidean_norms))
    sd_norm = float(np.std(euclidean_norms))

    return mean_norm + (3.0 * sd_norm)


def check_acc_abort_response(
    inlet: StreamInlet,
    acc_bases: dict,
    stim_direction: str,
    stim_onset: float,
    response,
    responded: bool,
    rt,
    trial_type: str,
    abort_intime: bool,
    window_ms: float = 100,
    verbose: bool = False
):
    """
    Pull buffered ACC samples and detect movement on the stimulated body side.

    Reads all samples currently in the LSL buffer (non-blocking), retains
    only those within the last `window_ms` ms, computes the RMS amplitude
    of the channel matching `stim_direction`, and flags a response when
    RMS > `threshold`.

    Parameters
    ----------
    inlet : StreamInlet
        Connected pylsl StreamInlet for the ACC stream.
    stim_direction : str
        'left' or 'right' — determines which ACC channel is evaluated.
    stim_onset : float
        time.time() value at stimulus onset, used to compute RT.
    response, responded, rt
        Current response state (passed through unchanged if no detection).
    trial_type : str
        'go', 'nogo', or 'abort'.
    abort_intime : bool
        True  → currently in the go phase of an abort trial.
        False → currently in the nogo (inhibition) phase.
    threshold : float
        RMS amplitude threshold for detection.
    window_ms : float
        Length of the analysis window in ms (default 100).
    ch_left, ch_right : int
        Channel indices for the left / right body-side ACC signal.

    Returns
    -------
    response, rt, responded  (same contract as check_response_keys)
    """

    if responded:
        return response, rt, responded

    # non-blocking pull of all buffered samples
    samples, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=250)
    if not samples:
        return response, rt, responded

    samples    = np.array(samples)     # (n_samples, n_channels)
    timestamps = np.array(timestamps)

    # keep only the last window_ms ms
    win_mask = timestamps >= (timestamps[-1] - window_ms / 1000.0)
    recent = samples[win_mask]
    if recent.shape[0] == 0:
        return response, rt, responded

    # pick the channel matching the stimulated body side
  
    samples = recent[:, ACC_LSL_IDX[stim_direction]]
    samples = samples - np.mean(samples, axis=0)  # detrend by removing mean of raw values (pos and neg)
    rms = np.sqrt(np.mean(samples ** 2))

    if rms > acc_bases[stim_direction] * 6:  # example threshold: 1.5x baseline; adjust as needed
        responded = True
        rt = time.time() - stim_onset

        if trial_type == 'go':
            response = 'correct'
        elif trial_type == 'nogo':
            response = 'incorrect'
        else:  # abort
            if abort_intime:
                response = 'correctIntime'
            else:
                response = 'incorrectOvertime'
    
        if verbose:
            print(f"ACC response detected ({response})! RMS: {rms:.2f}, RT: {rt:.3f} s")

    return response, rt, responded

### CHECK IF IT'S ALL HARD CODED