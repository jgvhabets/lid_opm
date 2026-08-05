import random
import csv
import time
import pygame
import os
from datetime import datetime

from tasks.trial import run_trial
from utils.check_acc_response import create_acc_inlet
from utils.response_inhibition_tracker import ResponseInhibitionTracker
from utils.lsl_stream import send_marker
import tasks.arduino_trigger as ard_trigger

random.seed(27)


def generate_trials(cfg):
    """Generate list of 'go', 'nogo', 'abort' trials based on proportions."""

    n_go = int(cfg["n_trials"] * cfg["go_proportion"])
    n_nogo = int(cfg["n_trials"] * cfg["nogo_proportion"])
    n_abort = cfg["n_trials"] - n_go - n_nogo
    trials = ["go"] * n_go + ["nogo"] * n_nogo + ["abort"] * n_abort

    random.seed(27)
    random.shuffle(trials)

    return trials


def jittered_iti(cfg):

    return random.uniform(cfg["iti_mean"] - cfg["iti_jitter"], cfg["iti_mean"] + cfg["iti_jitter"])


def waiting_screen(screen, clock, cfg):

    font = pygame.font.SysFont(None, 48)
    text = font.render("Wait for the task to start      (start collection, then press SPACE)",
                       True, (255, 255, 255))
    text_rect = text.get_rect(center=(cfg["screen_width"]//2, cfg["screen_height"]//2))

    waiting = True
    while waiting:
        screen.fill((0, 0, 0))
        screen.blit(text, text_rect)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False

        clock.tick(30)


def run_experiment(screen, cfg, clock, outlet=None, verbose=False,):
    """
    Run experiment, go vs no-go, vs go-abort.
    including a dynamic adjustment of the go-abort time,
    so in the end circa 50% of the trials will be correct.
    """
    trials = generate_trials(cfg)
    # create lists per trial type, with trial_directions, to later distribute directions
    # divide 50/50 over left and right
    random.seed(27)
    trial_directions = {
        'go': random.choices(["left", "right"], weights=[0.5, 0.5], k=trials.count("go")),
        'nogo': random.choices(["left", "right"], weights=[0.5, 0.5], k=trials.count("nogo")),
        'abort': random.choices(["left", "right"], weights=[0.5, 0.5], k=trials.count("abort"))
    }

    results = []
    stop_trial_count = 0

    exp_duration = cfg["experiment_duration"]
    stop_tracker = ResponseInhibitionTracker(
        initial_ssd_ms=cfg["abort_go_duration"] * 1000.0,
        step_size_ms=cfg["abort_step_size"] * 1000.0,
        min_ssd_ms=cfg.get("abort_min_duration", 0.15) * 1000.0,
        max_ssd_ms=cfg.get("abort_max_duration", 1.0) * 1000.0,
        stop_window_size=cfg.get("stop_window_size", 15),
        convergence_sd_ms=cfg.get("convergence_sd_ms", 30.0),
        go_rt_limit_ms=cfg.get("go_rt_limit_ms", 800.0),
    )

    # prepare log folder, __file__ is something like .../code/repo_root/gonogo_task/experiment.py
    gonogo_task_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(gonogo_task_dir)        # .../code/repo_root -> takes "code" folder
    parent_dir = os.path.dirname(parent_dir)        # .../ -> takes parent folder
    data_dir = os.path.join(os.path.dirname(parent_dir), "data", "gonogo_testdata")
    os.makedirs(data_dir, exist_ok=True)

    timestring = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_filename = cfg.get("log_file", f"gonogo_results_{timestring}.csv")
    log_path = os.path.join(data_dir, log_filename)

    # INIT ADRUINO
    if cfg['USE_ARDUINO']:
        TRIGGER_PIN, ARDUINO_BOARD = ard_trigger.init_board(port=cfg.get('ARDUINO_PORT'))
    else:
        TRIGGER_PIN, ARDUINO_BOARD = None, None

    # create ACC inlet if we need ACC-based feedback or adaptive abort timing
    if "abort" in trials and (cfg['check_correct_dtype'] == 'acc' or cfg.get('ADAPT_ABORT_TIME')):
        lsl_inlet, acc_bases = create_acc_inlet()
        print("Connected to ACC LSL stream for abort trial feedback.")
    else:
        lsl_inlet, acc_bases = None, None

    ### Waiting screen before starting task
    send_marker(outlet, f"TASK_INIT_beforeWaitScreen")

    waiting_screen(screen, clock, cfg)
    
    send_marker(outlet, f"TASK_START_afterWaitScreen")

    exp_start = time.time()  # take time after waiting screen as experiment starts


    for t, trial_type in enumerate(trials):
        # check whether time has expired
        if exp_duration and (time.time() - exp_start) >= exp_duration:
            print("Experiment duration reached, stopping early.")
            break
        # get direction for this trial and remove it from the list (so next time it will be different, but still balanced overall)
        trial_direction = trial_directions[trial_type].pop()

        send_marker(outlet, f"TRIAL_START_{t+1}_{trial_type}_{trial_direction}")

        # if trial type is abort, insert true acc-inlet, otherwise pass None to save resources in trial loop
        if trial_type == 'abort':
            use_acc_inlet = lsl_inlet
            abort_go_duration = stop_tracker.get_current_ssd_ms() / 1000.0
        else:
            use_acc_inlet = None
            abort_go_duration = None

        trial_data = run_trial(screen, trial_type, cfg, clock, outlet,
                               abort_go_duration=abort_go_duration,
                               trial_direction=trial_direction,
                               TRIGGER_PIN=TRIGGER_PIN,
                               acc_inlet=use_acc_inlet,
                               acc_bases=acc_bases,
                               verbose=verbose,)
        trial_data["trial"] = t + 1
        trial_data["timestamp"] = time.time() - exp_start
        results.append(trial_data)

        # --- adaptive staircase for abort ---
        if trial_type == "abort" and cfg['ADAPT_ABORT_TIME']:
            stop_trial_count += 1
            move_threshold = cfg.get("move_threshold")
            abort_acc_summary = trial_data.get("abort_acc_summary")

            if verbose:
                print(f'current time: {abort_go_duration}')
                print(f'abort ACC summary: {abort_acc_summary}')
                print(f'move threshold: {move_threshold}')

            stop_update = stop_tracker.record_stop_trial_from_summary(
                move_summary=abort_acc_summary,
                move_threshold=move_threshold,
            )
            trial_data["next_abort_go_duration"] = stop_update.next_ssd_ms / 1000.0
            trial_data["abort_update_decision"] = stop_update.decision
            trial_data["stop_success"] = stop_update.stop_success

            if verbose:
                print(f'next time: {stop_update.next_ssd_ms / 1000.0}')
                print(f'decision: {stop_update.decision}')
                print(f'stop success: {stop_update.stop_success}')
            
        if trial_type == 'abort':
            print(f'adjusted time: {stop_tracker.get_current_ssd_ms() / 1000.0}')

        target_stop_trials = cfg.get("target_stop_trials")
        stop_limit_reached = target_stop_trials is not None and stop_trial_count >= target_stop_trials
        if trial_type == 'abort' and (stop_tracker.has_converged() or stop_limit_reached):
            print("STOP tracker converged or target stop count reached. Stopping session early.")
            break

        send_marker(outlet, f"TRIAL_END_{t+1}_{trial_type}_{trial_direction}")

        iti = jittered_iti(cfg)
        pygame.time.wait(int(iti * 1000))


    ### end of experiment
    if results:
        stop_trials = [trial for trial in results if trial["trial_type"] == "abort"]
        successful_stops = [trial for trial in stop_trials if trial.get("stop_success") is True]
        failed_stop_rts = [trial["rt"] for trial in stop_trials if trial.get("stop_success") is False and trial.get("rt") is not None]
        go_rts = [trial["rt"] for trial in results if trial["trial_type"] == "go" and trial.get("rt") is not None]

        total_stop_trials = len(stop_trials)
        inhibition_accuracy = (len(successful_stops) / total_stop_trials) * 100 if total_stop_trials else 0.0
        mean_failed_stop_rt = sum(failed_stop_rts) / len(failed_stop_rts) if failed_stop_rts else float("nan")
        mean_go_rt = sum(go_rts) / len(go_rts) if go_rts else float("nan")
        horse_race_check = bool(mean_failed_stop_rt < mean_go_rt) if not (mean_failed_stop_rt != mean_failed_stop_rt or mean_go_rt != mean_go_rt) else False
        final_ssd_ms = stop_tracker.get_current_ssd_ms()
        session_valid = stop_tracker.has_converged() or (target_stop_trials is not None and stop_trial_count >= target_stop_trials)

        print(f"Session Valid: {session_valid} | Final SSD Plateau: {final_ssd_ms:.1f}ms | Accuracy: {inhibition_accuracy:.1f}%")
        print(f"Horse Race Check: {horse_race_check} | Mean Failed Stop RT: {mean_failed_stop_rt} | Mean Go RT: {mean_go_rt}")

    if cfg['USE_ARDUINO']:
        ard_trigger.close_board(pin=TRIGGER_PIN, board=ARDUINO_BOARD)

    if results:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    return results