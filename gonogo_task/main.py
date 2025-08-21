import pygame
import json

from gonogo_task.experiment import run_experiment


def main():
    """
    Run Go No-Go task from command line, working directory
    should be repo project root: lid_opm/

    cmd: python -m gonogo_task.main
    """
    # Load config
    with open("gonogo_task/config_gonogo.json", "r") as f:
        cfg = json.load(f)

    # Init pygame
    pygame.init()
    screen = pygame.display.set_mode((cfg["screen_width"], cfg["screen_height"]))
    pygame.display.set_caption("Go/No-Go Task")
    clock = pygame.time.Clock()

    # Run experiment
    results = run_experiment(screen, cfg, clock)
    print("Experiment finished. Results saved to", cfg["log_file"])

    pygame.quit()

if __name__ == "__main__":
    main()