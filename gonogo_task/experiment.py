import random
import csv
import time
import pygame
import os

from gonogo_task.trial import run_trial
from utils.lsl_stream import send_marker

random.seed(27)


def generate_trials(cfg):
    """Generate list of 'go', 'nogo', 'abort' trials based on proportions."""

    n_go = int(cfg["n_trials"] * cfg["go_proportion"])
    n_nogo = int(cfg["n_trials"] * cfg["nogo_proportion"])
    n_abort = cfg["n_trials"] - n_go - n_nogo
    trials = ["go"] * n_go + ["nogo"] * n_nogo + ["abort"] * n_abort

    random.shuffle(trials)

    return trials


def jittered_iti(cfg):

    return random.uniform(cfg["iti_mean"] - cfg["iti_jitter"], cfg["iti_mean"] + cfg["iti_jitter"])


def run_experiment(screen, cfg, clock, outlet=None):
    """
    Run experiment, go vs no-go, vs go-abort.
    including a dynamic adjustment of the go-abort time,
    so in the end circa 50% of the trials will be correct.
    """
    trials = generate_trials(cfg)
    results = []

    exp_start = time.time()
    exp_duration = cfg.get("experiment_duration", None)

    current_abort_duration = cfg["abort_go_duration"]

    # --- prepare log folder ---
    # __file__ is something like .../code/repo_root/gonogo_task/experiment.py
    gonogo_task_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(gonogo_task_dir)        # .../code/repo_root -> takes "code" folder
    parent_dir = os.path.dirname(parent_dir)        # .../ -> takes parent folder
    data_dir = os.path.join(os.path.dirname(parent_dir), "data", "gonogo_testdata")
    os.makedirs(data_dir, exist_ok=True)

    log_filename = cfg.get("log_file", "results.csv")
    log_path = os.path.join(data_dir, log_filename)


    for t, trial_type in enumerate(trials):
        if exp_duration and (time.time() - exp_start) >= exp_duration:
            print("Experiment duration reached, stopping early.")
            break

        send_marker(outlet, f"TRIAL_START_{t+1}_{trial_type}")

        trial_data = run_trial(screen, trial_type, cfg, clock, outlet,
                               abort_go_duration=current_abort_duration)
        trial_data["trial"] = t + 1
        trial_data["timestamp"] = time.time() - exp_start
        results.append(trial_data)

        # --- adaptive staircase for abort ---
        if trial_type == "abort":
            if trial_data["correct"]:
                current_abort_duration += cfg["abort_step_size"]
            else:
                current_abort_duration -= cfg["abort_step_size"]

            # keep inside safe bounds
            current_abort_duration = max(0.05,
                                         min(cfg["stimulus_duration"] - 0.05,
                                             current_abort_duration))

        send_marker(outlet, f"TRIAL_END_{t+1}")

        iti = jittered_iti(cfg)
        pygame.time.wait(int(iti * 1000))

    if results:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    return results