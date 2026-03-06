import pygame
import sys
import random
from .tank import Tank
from .bullet import Bullet
from .enemy import Enemy
from .obstacle import Obstacle
from .login import LoginScreen
from .constants import *


class Game:
    def __init__(self, username):
        # pygame already initialized by LoginScreen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tank Battle")
        self.clock = pygame.time.Clock()
        self.username = username
        self.tank = Tank(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 40)
        self.bullets = []  # player bullets
        self.enemy_bullets = []  # enemy bullets
        self.enemies = []
        self.obstacles = []
        self._create_obstacles()
        self.enemy_spawn_timer = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.font = pygame.font.Font(None, 36)

    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_SPACE:
                    bullet = Bullet(self.tank.rect.centerx, self.tank.rect.top)
                    self.bullets.append(bullet)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.tank.move(-TANK_SPEED)
        if keys[pygame.K_RIGHT]:
            self.tank.move(TANK_SPEED)
        if keys[pygame.K_UP]:
            self.tank.move(0, -TANK_SPEED)
        if keys[pygame.K_DOWN]:
            self.tank.move(0, TANK_SPEED)

    def _create_obstacles(self):
        # create connected wall structures
        wall_width = 50
        wall_height = 50
        
        # vertical wall on left
        for y in range(150, SCREEN_HEIGHT - 200, 60):
            self.obstacles.append(Obstacle(100, y, wall_width, wall_height))
        
        # vertical wall on right
        for y in range(150, SCREEN_HEIGHT - 200, 60):
            self.obstacles.append(Obstacle(SCREEN_WIDTH - 150, y, wall_width, wall_height))
        
        # horizontal walls in middle
        for x in range(300, SCREEN_WIDTH - 300, 70):
            self.obstacles.append(Obstacle(x, SCREEN_HEIGHT // 2 - 80, wall_width, wall_height))
            self.obstacles.append(Obstacle(x, SCREEN_HEIGHT // 2 + 80, wall_width, wall_height))
        
        # additional scattered obstacles for cover
        positions = [
            (400, 300), (600, 250), (800, 350),
            (300, 600), (800, 600), (1000, 300)
        ]
        for x, y in positions:
            self.obstacles.append(Obstacle(x, y, wall_width, wall_height))

    def _check_collisions(self):
        # player bullet vs enemy
        for bullet in self.bullets[:]:
            for enemy in self.enemies[:]:
                if bullet.rect.colliderect(enemy.rect):
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    self.enemies.remove(enemy)
                    self.score += 10
                    break
            # player bullet vs obstacle
            for obstacle in self.obstacles:
                if bullet.rect.colliderect(obstacle.rect):
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    break
        
        # enemy bullet vs player tank
        for bullet in self.enemy_bullets[:]:
            if bullet.rect.colliderect(self.tank.rect):
                if bullet in self.enemy_bullets:
                    self.enemy_bullets.remove(bullet)
                self.lives -= 1
                if self.lives <= 0:
                    self.game_over = True
        
        # enemy bullet vs player bullet (collision)
        for pbullet in self.bullets[:]:
            for ebullet in self.enemy_bullets[:]:
                if pbullet.rect.colliderect(ebullet.rect):
                    if pbullet in self.bullets:
                        self.bullets.remove(pbullet)
                    if ebullet in self.enemy_bullets:
                        self.enemy_bullets.remove(ebullet)
                    break

        # enemy vs tank
        for enemy in self.enemies[:]:
            if enemy.rect.colliderect(self.tank.rect):
                self.enemies.remove(enemy)
                self.lives -= 1
                if self.lives <= 0:
                    self.game_over = True

    def update(self):
        if self.game_over:
            return

        # spawn enemies periodically
        self.enemy_spawn_timer += 1
        if self.enemy_spawn_timer > 90:
            x = random.randint(20, SCREEN_WIDTH - 20)
            self.enemies.append(Enemy(x, -30))
            self.enemy_spawn_timer = 0

        for enemy in self.enemies[:]:
            enemy.update()
            # enemy shoots
            if enemy.can_shoot():
                ebullet = Bullet(enemy.rect.centerx, enemy.rect.bottom)
                self.enemy_bullets.append(ebullet)
            if enemy.rect.top > SCREEN_HEIGHT:
                self.enemies.remove(enemy)

        # update player bullets
        for bullet in self.bullets[:]:
            bullet.update()
            if bullet.rect.bottom < 0:
                self.bullets.remove(bullet)
        
        # update enemy bullets
        for bullet in self.enemy_bullets[:]:
            bullet.rect.y += 8  # enemy bullets go down
            if bullet.rect.top > SCREEN_HEIGHT:
                self.enemy_bullets.remove(bullet)

        self._check_collisions()

    def draw(self):
        self.screen.fill(BLACK)
        self.tank.draw(self.screen)
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        for enemy in self.enemies:
            enemy.draw(self.screen)
        for bullet in self.bullets:
            bullet.draw(self.screen)
        for bullet in self.enemy_bullets:
            bullet.draw(self.screen)
        
        # draw UI
        score_text = self.font.render(f"Player: {self.username} | Score: {self.score}", True, WHITE)
        lives_text = self.font.render(f"Lives: {self.lives}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (SCREEN_WIDTH - 200, 10))
        
        if self.game_over:
            game_over_text = self.font.render("GAME OVER!", True, RED)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(game_over_text, text_rect)
        
        pygame.display.flip()


if __name__ == "__main__":
    login_screen = LoginScreen()
    username = login_screen.run()
    
    game = Game(username)
    game.run()


def start():
    """Entry point used by bundled executable and importers."""
    login_screen = LoginScreen()
    username = login_screen.run()
    game = Game(username)
    game.run()
