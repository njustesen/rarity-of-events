from vizdoom import *
from arguments import get_args


def add_game_vars(game):
    game.add_available_game_variable(GameVariable.AMMO0)
    game.add_available_game_variable(GameVariable.AMMO1)
    game.add_available_game_variable(GameVariable.AMMO2)
    game.add_available_game_variable(GameVariable.AMMO3)
    game.add_available_game_variable(GameVariable.AMMO4)
    game.add_available_game_variable(GameVariable.AMMO5)
    game.add_available_game_variable(GameVariable.AMMO6)
    game.add_available_game_variable(GameVariable.AMMO7)
    game.add_available_game_variable(GameVariable.AMMO8)
    game.add_available_game_variable(GameVariable.AMMO9)
    game.add_available_game_variable(GameVariable.WEAPON0)
    game.add_available_game_variable(GameVariable.WEAPON1)
    game.add_available_game_variable(GameVariable.WEAPON2)
    game.add_available_game_variable(GameVariable.WEAPON3)
    game.add_available_game_variable(GameVariable.WEAPON4)
    game.add_available_game_variable(GameVariable.WEAPON5)
    game.add_available_game_variable(GameVariable.WEAPON6)
    game.add_available_game_variable(GameVariable.WEAPON7)
    game.add_available_game_variable(GameVariable.WEAPON8)
    game.add_available_game_variable(GameVariable.WEAPON9)
    game.add_available_game_variable(GameVariable.POSITION_X)
    game.add_available_game_variable(GameVariable.POSITION_Y)
    game.add_available_game_variable(GameVariable.ON_GROUND)
    game.add_available_game_variable(GameVariable.KILLCOUNT)
    game.add_available_game_variable(GameVariable.DEATHCOUNT)
    game.add_available_game_variable(GameVariable.ARMOR)
    game.add_available_game_variable(GameVariable.FRAGCOUNT)
    game.add_available_game_variable(GameVariable.HEALTH)


def make_env(worker_id, config_file_path=None, visual=False, deathmatch=False, bots=7):
    print("Initializing doom environment", worker_id, "...")
    game = DoomGame()
    game.load_config(config_file_path)
    add_game_vars(game)
    if deathmatch:
        game.set_doom_map("map02")
        game.add_game_args("-host 1 -deathmatch +timelimit 1.0 "
                           "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
                           "+viz_respawn_delay 10 1")
        game.add_game_args("+name AI +colorset 0")
        game.set_mode(Mode.PLAYER)

    game.set_window_visible(visual)
    game.init()
    if deathmatch:
        for i in range(bots):
            game.send_game_command("addbot")
    return game

