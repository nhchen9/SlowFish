#!/usr/bin/env python

"""Creating chess puzzles for lichess.org"""

import argparse
import chess
import chess.uci
import chess.pgn
import logging
import os
import sys
from modules.fishnet.fishnet import stockfish_command
from modules.puzzle.puzzle import puzzle
from modules.bcolors.bcolors import bcolors
from modules.investigate.investigate import investigate
from modules.api.api import post_puzzle
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("threads", metavar="THREADS", nargs="?", type=int, default=4,
                    help="number of engine threads")
parser.add_argument("memory", metavar="MEMORY", nargs="?", type=int, default=2048,
                    help="memory in MB to use for engine hashtables")
parser.add_argument("--depth", metavar="DEPTH", nargs="?", type=int, default=1,
                    help="depth for stockfish analysis")
parser.add_argument("--quiet", dest="loglevel",
                    default=logging.DEBUG, action="store_const", const=logging.INFO,
                    help="substantially reduce the number of logged messages")
parser.add_argument("--games", metavar="GAMES", default="../lichess.pgn",
                    help="A specific pgn with games")
parser.add_argument("--strict", metavar="STRICT", default=True,
                    help="If False then it will be generate more tactics but maybe a little ambiguous")
settings = parser.parse_args()
try:
    # Optionally fix colors on Windows and in journals if the colorama module
    # is available.
    import colorama
    wrapper = colorama.AnsiToWin32(sys.stdout)
    if wrapper.should_wrap():
        sys.stdout = wrapper.stream
except ImportError:
    pass

logging.basicConfig(format="%(message)s", level=settings.loglevel, stream=sys.stdout)
logging.getLogger("requests.packages.urllib3").setLevel(logging.WARNING)
logging.getLogger("chess.uci").setLevel(logging.WARNING)

engine = chess.uci.popen_engine(stockfish_command())
engine.setoption({'Threads': settings.threads, 'Hash': settings.memory})
engine.uci()
info_handler = chess.uci.InfoHandler()
engine.info_handlers.append(info_handler)

all_games = open(settings.games, "r")
tactics_file = open("tactics.pgn", "w")
game_id = 0
while True:
    game = chess.pgn.read_game(all_games)
    if game == None:
        break
    node = game

    game_id = game_id + 1

    prev_score = chess.uci.Score(None, None)
    puzzles = []


    engine.ucinewgame()

    while not node.is_end():
        next_node = node.variation(0)
        engine.position(next_node.board())

        engine.go(depth=settings.depth)
        cur_score = info_handler.info["score"][1]

        if numpy.random.random() < .1:
            puzzles.append(puzzle(node.board(), next_node.move, str(game_id), engine, info_handler, game, settings.strict))

        prev_score = cur_score
        node = next_node

    for i in puzzles:
        i.generate(settings.depth)
        if i.is_complete():
            puzzle_pgn = post_puzzle(i)
            tactics_file.write(puzzle_pgn)
            tactics_file.write("\n\n")

tactics_file.close()
