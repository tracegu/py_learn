import pygame
import sys
from .constants import *
from .user_manager import login_user, register_user


class LoginScreen:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tank Battle - Login")
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.Font(None, 72)
        self.font_large = pygame.font.Font(None, 48)
        self.font_normal = pygame.font.Font(None, 36)
        
        self.username_input = ""
        self.password_input = ""
        self.current_field = "username"  # username or password
        self.mode = "login"  # login or register
        self.message = ""
        self.message_timer = 0
        self.current_user = None

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_TAB:
                    # Switch field
                    self.current_field = "password" if self.current_field == "username" else "username"
                elif event.key == pygame.K_RETURN:
                    self.submit()
                elif event.key == pygame.K_BACKSPACE:
                    if self.current_field == "username":
                        self.username_input = self.username_input[:-1]
                    else:
                        self.password_input = self.password_input[:-1]
                elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    # Ctrl+S to switch mode
                    self.mode = "register" if self.mode == "login" else "login"
                    self.message = ""
                elif event.unicode.isprintable():
                    if self.current_field == "username":
                        if len(self.username_input) < 20:
                            self.username_input += event.unicode
                    else:
                        if len(self.password_input) < 20:
                            self.password_input += event.unicode

    def submit(self):
        if not self.username_input or not self.password_input:
            self.message = "Please fill all fields"
            self.message_timer = 60
            return
        
        if self.mode == "login":
            success, msg = login_user(self.username_input, self.password_input)
            if success:
                self.current_user = self.username_input
                self.message = "Login successful!"
                self.message_timer = 30
            else:
                self.message = msg
                self.message_timer = 60
        else:
            success, msg = register_user(self.username_input, self.password_input)
            if success:
                self.message = "Account created! Please login"
                self.message_timer = 60
                self.mode = "login"
                self.username_input = ""
                self.password_input = ""
            else:
                self.message = msg
                self.message_timer = 60

    def update(self):
        if self.message_timer > 0:
            self.message_timer -= 1

    def draw(self):
        self.screen.fill(BLACK)
        
        # Title
        title = self.font_title.render("TANK BATTLE", True, GREEN)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 80))
        self.screen.blit(title, title_rect)
        
        # Mode indicator
        mode_text = self.font_normal.render(f"[{self.mode.upper()}]", True, YELLOW)
        mode_rect = mode_text.get_rect(center=(SCREEN_WIDTH // 2, 180))
        self.screen.blit(mode_text, mode_rect)
        
        # Username label and input
        username_label = self.font_normal.render("Username:", True, WHITE)
        self.screen.blit(username_label, (300, 280))
        username_color = YELLOW if self.current_field == "username" else WHITE
        username_box = pygame.Rect(500, 280, 300, 50)
        pygame.draw.rect(self.screen, username_color, username_box, 2)
        username_text = self.font_normal.render(self.username_input, True, username_color)
        self.screen.blit(username_text, (510, 290))
        
        # Password label and input
        password_label = self.font_normal.render("Password:", True, WHITE)
        self.screen.blit(password_label, (300, 380))
        password_color = YELLOW if self.current_field == "password" else WHITE
        password_box = pygame.Rect(500, 380, 300, 50)
        pygame.draw.rect(self.screen, password_color, password_box, 2)
        password_display = "*" * len(self.password_input)
        password_text = self.font_normal.render(password_display, True, password_color)
        self.screen.blit(password_text, (510, 390))
        
        # Message
        if self.message:
            msg_color = GREEN if "success" in self.message else RED
            message_text = self.font_normal.render(self.message, True, msg_color)
            message_rect = message_text.get_rect(center=(SCREEN_WIDTH // 2, 500))
            self.screen.blit(message_text, message_rect)
        
        # Instructions
        instructions = [
            "TAB: Switch field | ENTER: Submit | Ctrl+S: Toggle Mode",
            f"Press ENTER to {self.mode}"
        ]
        for i, instruction in enumerate(instructions):
            instr_text = self.font_normal.render(instruction, True, GRAY)
            self.screen.blit(instr_text, (SCREEN_WIDTH // 2 - 400, SCREEN_HEIGHT - 150 + i * 50))
        
        pygame.display.flip()

    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
            
            if self.current_user:
                return self.current_user


# Add YELLOW and GRAY to constants imports
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
