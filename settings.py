import os
import retro


def install_games_from_rom_dir(romdir):
    """
    Add the ROMs to the Retro Game list
    """
    roms = [os.path.join(romdir, rom) for rom in os.listdir(romdir)]
    retro.data.merge(*roms, quiet=False)
