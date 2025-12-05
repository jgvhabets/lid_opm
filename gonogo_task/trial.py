import pygame
import time

from gonogo_task.stimuli import (
    draw_fixation, draw_go_stimulus, draw_nogo_stimulus
)
from utils.lsl_stream import send_marker

from arduino_trigger import send_trigger


def run_trial(screen, trial_type, cfg, clock, outlet=None,
              abort_go_duration=None,
              CHECKING_FREQ_FrameSec: int = 100,
              verbose = False,
              TRIGGER_PIN=None,):
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
    pygame.time.wait(500)  # additional to ITI

    # --- Stimulus onset ---
    stim_onset = time.time()
    responded = False

    ### SEND ARDUINO TRIGGER
    send_trigger(pin=TRIGGER_PIN, TRIG_type=trial_type,)


    if trial_type == "go":
        # Show green circle for entire duration
        screen.fill(cfg["bg_color"])

        direction = draw_go_stimulus(screen, cfg["stimulus_color"],
                                 cfg["screen_width"], cfg["screen_height"],
                                 cfg["arrow_size"])
        pygame.display.flip()
        send_marker(outlet, f"STIM_ONSET_go_{direction}")

        if verbose: print(f'\n\nSTART {trial_type}, direction: {direction}')

        while (time.time() - stim_onset < cfg["stimulus_duration"]) and not responded:
            # only check when not responded yet:
            response, rt, responded = check_response(cfg, stim_onset, direction,
                                                     responded, outlet, trial_type,
                                                     FEEDBACK_TYPE=cfg['check_correct_dtype'],
                                                     verbose=verbose,)
            clock.tick(CHECKING_FREQ_FrameSec)

     

    elif trial_type == "nogo":
        # Show red square for entire duration
        screen.fill(cfg["bg_color"])
        direction = draw_nogo_stimulus(screen, cfg["nogo_stimulus_color"],
                                       cfg["screen_width"], cfg["screen_height"],
                                       cfg["arrow_size"])
        pygame.display.flip()
        send_marker(outlet, f"STIM_ONSET_nogo_{direction}")
        
        if verbose: print(f'\n\nSTART {trial_type}, direction: {direction}')

        while (time.time() - stim_onset < cfg["stimulus_duration"]) and not responded:
            # only check when not responded yet:
            response, rt, responded = check_response(cfg, stim_onset, direction,
                                                     responded, outlet, trial_type,
                                                     FEEDBACK_TYPE=cfg['check_correct_dtype'],
                                                     verbose=verbose,)
            clock.tick(CHECKING_FREQ_FrameSec)



    elif trial_type == "abort":
        # Phase 1: green go (abort_go_duration)
        screen.fill(cfg["bg_color"])
        direction = draw_go_stimulus(screen, cfg["stimulus_color"],
                                 cfg["screen_width"], cfg["screen_height"],
                                 cfg["arrow_size"])
        pygame.display.flip()
        send_marker(outlet, f"STIM_ONSET_abort_go_{direction}")
        
        if verbose: print(f'\n\nSTART {trial_type}, direction: {direction}')

        while (time.time() - stim_onset < abort_go_duration) and not responded:
            # only check when not responded yet:
            if verbose: print('.........check abort\tinTIME')
            response, rt, responded = check_response(cfg, stim_onset, direction,
                                                     responded, outlet, trial_type,
                                                     abort_intime=True,
                                                     FEEDBACK_TYPE=cfg['check_correct_dtype'],
                                                     verbose=verbose,)
            clock.tick(CHECKING_FREQ_FrameSec)

        # Phase 2: switch to red nogo
        if not responded:
            # do not go into Phase 2 if sub responded already
            screen.fill(cfg["bg_color"])
            draw_nogo_stimulus(screen, cfg["nogo_stimulus_color"],
                        cfg["screen_width"], cfg["screen_height"],
                        cfg["arrow_size"], direction=direction)
            pygame.display.flip()
            send_marker(outlet, f"STIM_ONSET_abort_nogo_{direction}")
            
            if verbose: print(f"START PHASE222 abort_nogo_{direction}")

            while (time.time() - stim_onset < cfg["stimulus_duration"]) and not responded:
                # only check when not responded yet
                if verbose: print('.........check abort\tOVERTIME')
                
                response, rt, responded = check_response(cfg, stim_onset, direction,
                                                        responded, outlet, trial_type,
                                                        abort_intime=False,
                                                        FEEDBACK_TYPE=cfg['check_correct_dtype'],
                                                        verbose=verbose,)
                clock.tick(CHECKING_FREQ_FrameSec)


    # interpret correctness of response
    if verbose: print('collected response:', response)
    
    CORRECT_ABORT = None
    if trial_type == 'abort' and cfg['ADAPT_ABORT_TIME']:
        if type(response) == str:
            if 'correct' in response and not 'incorrect' in response:
                if 'overtime' in response.lower():
                    CORRECT_ABORT = False
                else:
                    CORRECT_ABORT = True
            else:
                CORRECT_ABORT = False



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
        "CORRECT_ABORT": CORRECT_ABORT
    }


def check_response(cfg, stim_onset, stim_direction,
                   responded, outlet, trial_type,
                   abort_intime: bool = True,
                   FEEDBACK_TYPE: str = 'none',
                   verbose=False,):
    
    """
    Helper to check for keypress during stimulus display.
    """
    
    FEEDBACK_TYPE = FEEDBACK_TYPE.lower()

    allowed_dtypes = ['none', 'keys', 'acc']
    
    assert FEEDBACK_TYPE in allowed_dtypes, (
        f'correctness feedback datatype given ("{FEEDBACK_TYPE}") not'
        f' allowed; should be in {allowed_dtypes}.'
        f'\ncheck "check_correct_dtype" in config_gonogo.json'
    )

    response = None
    rt = None

    for event in pygame.event.get():
    
        if FEEDBACK_TYPE == 'keys':
            response, rt, responded = check_response_keys(
                event, response, responded, stim_onset, rt,
                stim_direction, trial_type, abort_intime
            )
            
        if responded:
            send_marker(outlet, f"RESPONSE_{trial_type}_{response}_RT={rt:.3f}")
            
            if verbose: print(f"freshly CATCHED RESPONSE_{trial_type}_{response}_RT={rt:.3f}")

    return response, rt, responded


def check_response_keys(event, response, responded, stim_onset, rt,
                        stim_direction, trial_type, abort_intime):

    if event.type == pygame.KEYDOWN and not responded:
        
        if event.key == pygame.K_LEFT:
            response = 'left'
            responded = True
        
        elif event.key == pygame.K_RIGHT:
            response = 'right'
            responded = True
        
        

        
    if responded:
        
        rt = time.time() - stim_onset
        
        if trial_type == 'go':
            if stim_direction.lower() == response:
                response = 'correct'
            else:
                response = 'incorrect'

        elif trial_type == 'nogo':
            response = 'incorrect'
        
        else:  # trial_type == abort
            if stim_direction.lower() == response:
                response = 'correct'
            else:
                response = 'incorrect'
            # responded within go phase of abort
            if abort_intime:
                response += 'Intime'
            else:
                response += 'Overtime'

    
    return response, rt, responded