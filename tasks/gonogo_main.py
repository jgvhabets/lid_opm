import pygame
import json
import os

from tasks.gonogo_experiment import run_experiment
from utils.lsl_stream import create_lsl_outlet

"""

Variables in config.json:
- screen_id: 0 is first computer screen, 1 is possible external monitor.
"""


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config_gonogo.json")

    with open(config_path, "r") as f:

        return json.load(f)


def main():
    """
    Run Go No-Go task from command line, working directory
    should be repo project root: lid_opm/

    cmd: python -m tasks.gonogo_main
    """
    # Init pygame
    pygame.init()

    # load config
    cfg = load_config()

    # detect available displays
    num_displays = pygame.display.get_num_displays()
    screen_id = min(cfg.get("screen_id", 0), num_displays - 1)

    # get system display resolution
    display_info = pygame.display.Info()
    # screen_width, screen_height = display_info.current_w, display_info.current_h

    # create fullscreen window
    screen = pygame.display.set_mode(
        (0, 0),  # let pygame pick fullscreen resolution for that display
        pygame.FULLSCREEN,
        display=screen_id
    )
    # get actual resolution of this window
    screen_width, screen_height = screen.get_size()

    # overwrite width/height dynamically
    cfg["screen_width"] = screen_width
    cfg["screen_height"] = screen_height

    # optional: black background
    screen.fill((0, 0, 0))
    pygame.display.flip()

    # Set caption of window
    pygame.display.set_caption("Go/No-Go Task")

    # init LSL stream
    outlet = create_lsl_outlet()

    # Run experiment
    clock = pygame.time.Clock()
    run_experiment(screen, cfg, clock, outlet,)
    
    print("Experiment finished. Results saved to", cfg["log_file"])

    pygame.quit()

if __name__ == "__main__":
    main()