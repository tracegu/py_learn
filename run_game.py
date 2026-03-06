"""Tiny launcher that calls the package entrypoint.

This is used as the script PyInstaller bundles. Importing the package
and calling `start()` is more robust inside a frozen executable than
using runpy.run_module.
"""

from tank_game import main


if __name__ == '__main__':
    main.start()
