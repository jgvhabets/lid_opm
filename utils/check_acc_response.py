import time
import numpy as np
from pylsl import StreamInlet, resolve_stream


def create_acc_inlet(stream_name: str = 'acc', timeout: float = 5.0) -> StreamInlet:
    """Resolve an ACC LSL stream and return a connected StreamInlet.

    Call this once before starting the task loop and pass the returned
    inlet to run_trial() via the acc_inlet parameter.
    """
    streams = resolve_stream('name', stream_name, timeout=timeout)
    if not streams:
        raise RuntimeError(
            f"No LSL stream named '{stream_name}' found within {timeout} s. "
            "Make sure the ACC stream is broadcasting before starting the task."
        )
    return StreamInlet(streams[0], max_buflen=5)


def check_acc_abort_response(
    inlet: StreamInlet,
    stim_direction: str,
    stim_onset: float,
    response,
    responded: bool,
    rt,
    trial_type: str,
    abort_intime: bool,
    threshold: float,
    window_ms: float = 100,
    ch_left: int = 0,
    ch_right: int = 1,
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
    samples, timestamps = inlet.pull_chunk(timeout=0.0)
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
    ch_idx = ch_left if stim_direction.lower() == 'left' else ch_right
    rms = np.sqrt(np.mean(recent[:, ch_idx] ** 2))

    if rms > threshold:
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

    return response, rt, responded
