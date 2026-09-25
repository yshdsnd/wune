import io
import struct
import unittest

import pygame as pg

from wune.icons import ASSETS, pygame_icon, set_tk_icon


class IconTests(unittest.TestCase):
    def test_ico_has_decodable_rgba_images_at_all_windows_sizes(self):
        data = (ASSETS / "Wune.ico").read_bytes()
        reserved, kind, count = struct.unpack_from("<HHH", data)
        self.assertEqual((reserved, kind, count), (0, 1, 8))
        sizes = []
        for index in range(count):
            width, height, _, _, planes, bits, length, offset = struct.unpack_from(
                "<BBBBHHII", data, 6 + index * 16)
            size = width or 256
            self.assertEqual(size, height or 256)
            self.assertEqual((planes, bits), (1, 32))
            self.assertLessEqual(offset + length, len(data))
            surface = pg.image.load(io.BytesIO(data[offset:offset + length]), "icon.png")
            self.assertEqual(surface.get_size(), (size, size))
            # Lanczos may leave an almost transparent rounding fringe.
            self.assertLessEqual(surface.get_at((0, 0)).a, 2)
            # Both low-level green and peak red remain visible even at 16 px.
            pixels = [surface.get_at((x, y)) for x in range(size) for y in range(size)]
            self.assertTrue(any(g > 130 and g > r * 1.5 and g > b * 1.5 and a > 200
                                for r, g, b, a in pixels), size)
            self.assertTrue(any(r > 130 and r > g * 1.5 and r > b * 1.5 and a > 200
                                for r, g, b, a in pixels), size)
            sizes.append(size)
        self.assertEqual(sorted(sizes), [16, 24, 32, 48, 64, 96, 128, 256])

    def test_window_icon_is_independent_of_working_directory(self):
        self.assertTrue(ASSETS.is_absolute())
        icon = pygame_icon()
        self.assertEqual(icon.get_size(), (64, 64))
        self.assertEqual(icon.get_at((0, 0)).a, 0)

    def test_tk_accepts_icon_and_child_window_inherits_default(self):
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        try:
            set_tk_icon(root)
            child = tk.Toplevel(root)
            child.withdraw()
            root.update_idletasks()
            child.destroy()
        finally:
            root.destroy()
