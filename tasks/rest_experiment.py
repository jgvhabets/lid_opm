import random
import time
import pygame
import numpy as np

from utils.lsl_stream import send_marker
from tasks.stimuli import draw_fixation
import tasks.arduino_trigger as ard_trigger

random.seed(27)


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



def run_experiment(screen, cfg, clock, outlet=None,):
    """
    Run experiment, rest with fixation point.
    """
    
    exp_duration = cfg["experiment_duration"]

    synctrigger_interval = cfg["synctrigger_interval"]  # in seconds

    n_triggers = exp_duration / synctrigger_interval

    # INIT ADRUINO
    if cfg['USE_ARDUINO']:
        TRIGGER_PIN, ARDUINO_BOARD = ard_trigger.init_board()
    else:
        TRIGGER_PIN, ARDUINO_BOARD = None, None

    ### Waiting screen before starting task
    send_marker(outlet, f"TASK_INIT_beforeWaitScreen")

    waiting_screen(screen, clock, cfg)

    send_marker(outlet, f"REST_START_afterWaitScreen")

    exp_start = time.time()

    for i in np.arange(n_triggers + 1):
        # print(f'sync trigger round {i + 1}')

        if exp_duration and (time.time() - exp_start) >= exp_duration:
            print("Experiment duration reached, stopping early.")
            break

        send_marker(outlet, f"syncInterval_START_{i+1}")

        screen.fill(cfg["bg_color"])
        draw_fixation(screen, cfg["fixation_color"], cfg["screen_width"], cfg["screen_height"])
        pygame.display.flip()

        # send sync trigger
        # print(f'SEND trigger round {i + 1}')
        send_marker(outlet, f"sendTrigger_{i+1}")
        
        if cfg['USE_ARDUINO']:
            ard_trigger.send_trigger(pin=TRIGGER_PIN, TRIG_type='rest',)

        # wait for next sync trigger
        pygame.time.wait(synctrigger_interval * 1000)  # in milliseconds

        send_marker(outlet, f"syncInterval_END_{i+1}")
        # print(f'### END trigger round {i + 1}')


    ### end of experiment
    if cfg['USE_ARDUINO']:
        ard_trigger.close_board(pin=TRIGGER_PIN, board=ARDUINO_BOARD)
    