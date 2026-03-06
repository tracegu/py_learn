import pygame
import os
import random
from .constants import *


class Enemy:
    def __init__(self, x, y, width=20, height=20):
        self.rect = pygame.Rect(0, 0, width, height)
        self.rect.center = (x, y)
        self.color = RED
        self.speed = random.randint(1, 3)
        self.move_direction = random.choice([-1, 1])  # -1 left, 1 right
        self.move_timer = random.randint(30, 120)  # time to change direction
        self.shoot_timer = random.randint(60, 120)  # time between shots
        
        # 尝试加载图片，如果失败则使用颜色
        try:
            asset_path = os.path.join(os.path.dirname(__file__), 'assets', 'enemy_tank.png')
            self.image = pygame.image.load(asset_path)
            self.image = pygame.transform.scale(self.image, (width, height))
            self.rect = self.image.get_rect(center=(x, y))
        except:
            self.image = None

    def update(self):
        # move downward
        self.rect.y += self.speed
        # move left and right
        self.rect.x += self.move_direction * 2
        
        # keep within screen width
        if self.rect.left < 0 or self.rect.right > 1200:
            self.move_direction *= -1
        
        # update direction timer
        self.move_timer -= 1
        if self.move_timer <= 0:
            self.move_direction = random.choice([-1, 1])
            self.move_timer = random.randint(30, 120)
        
        # update shoot timer
        self.shoot_timer -= 1

    def can_shoot(self):
        """Return True if enemy should shoot"""
        if self.shoot_timer <= 0:
            self.shoot_timer = random.randint(60, 120)
            return True
        return False

    def draw(self, surface):
        if self.image:
            surface.blit(self.image, self.rect)
        else:
            # fallback to shape drawing
            pygame.draw.rect(surface, self.color, self.rect)
            turret_rect = pygame.Rect(0, 0, self.rect.width // 2, self.rect.height // 2)
            turret_rect.center = self.rect.center
            pygame.draw.rect(surface, BLACK, turret_rect)
