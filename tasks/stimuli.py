import pygame
import random

random.seed(27)

def draw_fixation(screen, color, width, height):
    """Draw a fixation cross in the center."""
    pygame.draw.line(screen, color, (width//2 - 10, height//2),
                     (width//2 + 10, height//2), 2)
    pygame.draw.line(screen, color, (width//2, height//2 - 10),
                     (width//2, height//2 + 10), 2)


def draw_arrow(screen, color, width, height, size, direction):
    """Draws a left or right arrow with shaft + triangle head."""
    cx, cy = width // 2, height // 2
    shaft_length = size * 0.6
    shaft_thickness = size * 0.15
    head_length = size * 0.4
    head_width = size * 0.4

    if direction == "left":
        # shaft rectangle
        shaft_rect = pygame.Rect(
            cx - shaft_length//2, cy - shaft_thickness//2,
            shaft_length, shaft_thickness
        )
        pygame.draw.rect(screen, color, shaft_rect)

        # triangle head (points left)
        points = [
            (shaft_rect.left - head_length, cy),
            (shaft_rect.left, cy - head_width//2),
            (shaft_rect.left, cy + head_width//2),
        ]
        pygame.draw.polygon(screen, color, points)

    else:  # right
        # shaft rectangle
        shaft_rect = pygame.Rect(
            cx - shaft_length//2, cy - shaft_thickness//2,
            shaft_length, shaft_thickness
        )
        pygame.draw.rect(screen, color, shaft_rect)

        # triangle head (points right)
        points = [
            (shaft_rect.right + head_length, cy),
            (shaft_rect.right, cy - head_width//2),
            (shaft_rect.right, cy + head_width//2),
        ]
        pygame.draw.polygon(screen, color, points)



def draw_go_stimulus(screen, color, width, height, size, direction=None):
    if direction is None:
        direction = random.choice(["left", "right"])
    draw_arrow(screen, color, width, height, size, direction)

    return direction


def draw_nogo_stimulus(screen, color, width, height, size, direction=None):
    if direction is None:
        direction = random.choice(["left", "right"])
    draw_arrow(screen, color, width, height, size, direction)
    
    return direction