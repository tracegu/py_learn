import pygame
import os
from .constants import *


class Bullet:
    def __init__(self, x, y, width=5, height=10, color=WHITE):
        self.rect = pygame.Rect(0, 0, width, height)
        self.rect.center = (x, y)
        self.color = color
        
        # 尝试加载图片，如果失败则使用颜色
        try:
            asset_path = os.path.join(os.path.dirname(__file__), 'assets', 'bullet.png')
            self.image = pygame.image.load(asset_path)
            self.rect = self.image.get_rect(center=(x, y))
        except:
            self.image = None

    def update(self):
        self.rect.y += BULLET_SPEED

    def draw(self, surface):
        if self.image:
            surface.blit(self.image, self.rect)
        else:
            # fallback to shape drawing
            pygame.draw.rect(surface, self.color, self.rect)
