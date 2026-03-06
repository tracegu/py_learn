import pygame
import os
from .constants import *


class Obstacle:
    def __init__(self, x, y, width=50, height=50, color=BROWN):
        self.rect = pygame.Rect(0, 0, width, height)
        self.rect.topleft = (x, y)
        self.color = color
        
        # 尝试加载图片，如果失败则使用颜色
        try:
            asset_path = os.path.join(os.path.dirname(__file__), 'assets', 'obstacle.png')
            self.image = pygame.image.load(asset_path)
            self.rect = self.image.get_rect(topleft=(x, y))
        except:
            self.image = None

    def draw(self, surface):
        if self.image:
            surface.blit(self.image, self.rect)
        else:
            # fallback to shape drawing
            pygame.draw.rect(surface, self.color, self.rect)
