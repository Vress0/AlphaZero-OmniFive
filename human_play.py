# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
import torch
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure  # noqa: F401
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet  # Pytorch


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 5
    width, height = 8, 8
    model_file = 'best_policy.model'
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required to run human_play with PyTorch backend.")
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################
        if not model_file:
            raise ValueError("model_file must point to a PyTorch checkpoint saved by train.py")
        best_policy = PolicyValueNet(width, height,
                                     model_file=model_file,
                                     use_gpu=True)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model file '{model_file}' not found. Run train.py to produce a PyTorch checkpoint before starting human_play."
        )
    except (RuntimeError, pickle.UnpicklingError) as exc:
        raise RuntimeError(
            "Failed to load PyTorch model. Ensure the checkpoint was produced by train.py (PyTorch) rather than the legacy Theano model."
        ) from exc
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
