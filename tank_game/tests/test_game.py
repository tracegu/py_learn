import unittest

from tank_game.constants import SCREEN_WIDTH, SCREEN_HEIGHT, TANK_SPEED
from tank_game.tank import Tank
from tank_game.bullet import Bullet


class BasicTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(SCREEN_WIDTH, 800)
        self.assertEqual(SCREEN_HEIGHT, 600)
        self.assertTrue(TANK_SPEED > 0)

    def test_tank_creation(self):
        tank = Tank(100, 100)
        self.assertEqual(tank.rect.center, (100, 100))

    def test_bullet_movement(self):
        bullet = Bullet(50, 50)
        old_y = bullet.rect.y
        bullet.update()
        self.assertLess(bullet.rect.y, old_y)


if __name__ == '__main__':
    unittest.main()
