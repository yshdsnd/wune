"""Module entry point: python -m wune."""
import multiprocessing
import sys

import pygame as pg

from wune.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as ex:
        print("Error:", ex)
        pg.quit()
        sys.exit(1)
