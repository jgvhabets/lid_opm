import pygame

def draw_fixation(screen, color, screen_width, screen_height):
    """Draw a fixation cross in the center."""
    center_x, center_y = screen_width // 2, screen_height // 2
    pygame.draw.line(screen, color, (center_x - 10, center_y), (center_x + 10, center_y), 2)
    pygame.draw.line(screen, color, (center_x, center_y - 10), (center_x, center_y + 10), 2)

def draw_go_stimulus(screen, color, screen_width, screen_height):
    """Draw a 'GO' stimulus (green circle)."""
    center = (screen_width // 2, screen_height // 2)
    pygame.draw.circle(screen, color, center, 50)

def draw_nogo_stimulus(screen, color, screen_width, screen_height):
    """Draw a 'NO-GO' stimulus (red square)."""
    rect = pygame.Rect(0, 0, 100, 100)
    rect.center = (screen_width // 2, screen_height // 2)
    pygame.draw.rect(screen, color, rect)