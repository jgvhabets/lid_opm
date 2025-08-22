import pygame
import time

from gonogo_task.stimuli import (
    draw_fixation, draw_go_stimulus, draw_nogo_stimulus
)
from utils.lsl_stream import send_marker



def run_trial(screen, trial_type, cfg, clock, outlet=None,
              abort_go_duration=None,):
    """
    
    Abort trials: depending on successful or unsuccessful
        inhibition of preceding Go-Signal, the Abort-Signal
        is increased or decreased with X seconds (given as
        a parameter in the config, defaults 0.05 s; and the 
        Abort-Signal interval defaults to 250 ms; acc. to Cao 2024).
        
    
    """
    response = None
    rt = None

    # take default go-abort time at beginning
    if abort_go_duration is None:
        abort_go_duration = cfg["abort_go_duration"]

    # --- Fixation during ITI ---
    screen.fill(cfg["bg_color"])
    draw_fixation(screen, cfg["fixation_color"], cfg["screen_width"], cfg["screen_height"])
    pygame.display.flip()
    pygame.time.wait(500)  # TODO: why hardcoded

    # --- Stimulus onset ---
    stim_onset = time.time()
    responded = False


    if trial_type == "go":
        # Show green circle for entire duration
        screen.fill(cfg["bg_color"])

        direction = draw_go_stimulus(screen, cfg["stimulus_color"],
                                 cfg["screen_width"], cfg["screen_height"],
                                 cfg["arrow_size"])
        pygame.display.flip()
        send_marker(outlet, f"STIM_ONSET_go")

        while time.time() - stim_onset < cfg["stimulus_duration"]:
            response, rt, responded = check_response(cfg, stim_onset, responded, outlet, trial_type)
            clock.tick(60)

     
    elif trial_type == "nogo":
        # Show red square for entire duration
        screen.fill(cfg["bg_color"])
        direction = draw_nogo_stimulus(screen, cfg["nogo_stimulus_color"],
                                   cfg["screen_width"], cfg["screen_height"],
                                   cfg["arrow_size"])
        pygame.display.flip()
        send_marker(outlet, f"STIM_ONSET_nogo")

        while time.time() - stim_onset < cfg["stimulus_duration"]:
            response, rt, responded = check_response(cfg, stim_onset, responded, outlet, trial_type)
            clock.tick(60)


    elif trial_type == "abort":
        # Phase 1: green go (abort_go_duration)
        screen.fill(cfg["bg_color"])
        direction = draw_go_stimulus(screen, cfg["stimulus_color"],
                                 cfg["screen_width"], cfg["screen_height"],
                                 cfg["arrow_size"])
        pygame.display.flip()
        send_marker(outlet, f"STIM_ONSET_abort_go")

        while time.time() - stim_onset < cfg["abort_go_duration"]:
            response, rt, responded = check_response(cfg, stim_onset, responded, outlet, trial_type)
            clock.tick(60)

        # Phase 2: switch to red nogo
        screen.fill(cfg["bg_color"])
        draw_nogo_stimulus(screen, cfg["nogo_stimulus_color"],
                       cfg["screen_width"], cfg["screen_height"],
                       cfg["arrow_size"], direction=direction)
        pygame.display.flip()
        send_marker(outlet, f"STIM_ONSET_abort_nogo")

        while time.time() - stim_onset < cfg["stimulus_duration"]:
            response, rt, responded = check_response(cfg, stim_onset, responded, outlet, trial_type)
            clock.tick(60)


    # --- After stimulus: return to fixation (ITI) ---
    screen.fill(cfg["bg_color"])
    draw_fixation(screen, cfg["fixation_color"], cfg["screen_width"], cfg["screen_height"])
    pygame.display.flip()

    
    return {
        "trial_type": trial_type,
        "direction": direction,
        "response": response,
        "rt": rt,
        "abort_go_duration": abort_go_duration if trial_type == "abort" else None,
        "correct": (
            (trial_type == "go" and response == "pressed") or
            (trial_type == "nogo" and response is None) or
            (trial_type == "abort" and response is None)
        )
    }


def check_response(cfg, stim_onset, responded, outlet, trial_type):
    """Helper to check for keypress during stimulus display."""
    response = None
    rt = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); raise SystemExit
        if event.type == pygame.KEYDOWN:
            if pygame.key.name(event.key) in cfg["response_keys"] and not responded:
                response = "pressed"
                rt = time.time() - stim_onset
                responded = True
                send_marker(outlet, f"RESPONSE_{trial_type}_RT={rt:.3f}")
    return response, rt, responded