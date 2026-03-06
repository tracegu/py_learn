import pygame
import os
from .constants import *


class Tank:
    def __init__(self, x, y, width=None, height=None, color=GREEN):
        if width is None:
            width = TANK_SIZE
        if height is None:
            height = TANK_SIZE
        self.rect = pygame.Rect(0, 0, width, height)
        self.rect.center = (x, y)
        self.color = color
        
        # 尝试加载图片，如果失败则使用颜色
        try:
            asset_path = os.path.join(os.path.dirname(__file__), 'assets', 'player_tank.png')
            self.image = pygame.image.load(asset_path)
            self.image = pygame.transform.scale(self.image, (width, height))
            self.rect = self.image.get_rect(center=(x, y))
        except:
            self.image = None

    def move(self, dx=0, dy=0):
        self.rect.x += dx
        self.rect.y += dy
        # keep within screen
        self.rect.x = max(0, min(self.rect.x, SCREEN_WIDTH - self.rect.width))
        self.rect.y = max(0, min(self.rect.y, SCREEN_HEIGHT - self.rect.height))

    def draw(self, surface):
        if self.image:
            surface.blit(self.image, self.rect)
        else:
            # fallback to shape drawing
            pygame.draw.rect(surface, self.color, self.rect)
            turret_rect = pygame.Rect(0, 0, self.rect.width // 2, self.rect.height // 2)
            turret_rect.center = self.rect.center
            pygame.draw.rect(surface, BLACK, turret_rect)
