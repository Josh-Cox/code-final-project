import sys
import chess.pgn
import random
import copy
import re
import numpy as np

def find_number_moves(moves):
    """
    Finds the number of moves in a given game

    :param moves: pgn of moves for a game
    :return: number of moves in the game as an integer
    """
    
    return int(re.findall(" (\d+)\.", moves)[-1])

def get_random_pos(game):
    """
    Gets a random board position in a given game

    :param game: game to find position in
    :return: random board position
    """
    board = game.board()
    
    num_moves = find_number_moves(str(game.mainline_moves()))
    
    rand = random.randrange(1, num_moves)
    count = 0

    for move in game.mainline_moves():
        board.push(move)
        count += 1
        if count == rand:
            break;

    return board.fen()

def convert_to_bitboard(fen):
    """
    Converts a board position to bitboard representation

    :param board: board of given position to convert
    :return: bitboard array of position
    """

    # convert fen to board
    board = chess.Board(fen)

    # convert to string
    board = str(board)

    # define piece values
    piece_values = {
        '.': 0,
        'p': 1,
        'r': 2,
        'n': 3,
        'b': 4,
        'q': 5,
        'k': 6,
        'P': 7,
        'R': 8,
        'N': 9,
        'B': 10,
        'Q': 11,
        'K': 12,
        
    }
    
    def convert_to_value(x):
        """
        Converts a character to its respective integer value
    
        :param x: character to convert
        :return: integer values respetive to character
        """
        return piece_values[x]

    # vectorize function
    vconvert = np.vectorize(convert_to_value)

    # remove newlines and whitespace
    board = np.char.replace(board, ' ', '')
    board = np.char.replace(board, '\n', '')

    # map convert function to array
    board = np.array(list(map(vconvert, str(board))))
    
    return board

def find_turn(fen):
    """
    Finds which color's turn it is

    :param fen: fen of position
    :return: w (white) or b (black)
    """

    # set to black
    turn = 'b'

    # if white then set to white
    if 'w' in fen:
        turn = 'w'

    return turn
    
def find_kings(pos):
    """
    Finds the position of each king

    :param pos: position in bitboard representation 
    :return: position of king in bitboard array
    """

    # find index of each king
    w_king_pos = np.where(pos == 12)
    b_king_pos = np.where(pos == 6)

    return np.array([w_king_pos[0][0], b_king_pos[0][0]])

def check_top_rank(color, pos):
    # check if given position is in the opposite top rank
    if color == 'w':
        if pos <= 7:
            return True
        else:
            return False
    elif color == 'b':
        if pos >= 57:
            return True
        else:
            return False

def king_safety_eval(king_pos, method, bitboard):
    """
    Evaluates the safety of given king (cap of 3 pushes for each pawn)

    :param color: which color's king to evaluate
    :param king_pos: position of king to evaluate
    :param method: method to use (standard or exponential)
    :return: value to represent evaluation of king safety
    
    """
    safety = [1, 1] # higher = worse safety
    king_edge = ['N', 'N'] #(N for none, L for left, R for right, T for top)
    white_king = king_pos[0]
    black_king = king_pos[1]

    # ----- WHITE KING -----
    
    # check if king is on opposite top rank
    if check_top_rank('w', white_king):
        king_edge[0] = 'T'
    # check for position 0 (edge case to prevent division by 0)
    elif white_king == 0:
        king_edge[0] = 'L'
    # if king is at left edge
    elif white_king % 8 == 0:
        king_edge[0] = 'L'
    # if king is at right edge
    elif white_king % 8 == 7:
        king_edge[0] = 'R'

    # Determine pawn positions
    
    def check_pawns_in_file(color, checking_pos):
        points = 0
        
        if color == 'w':
            pawn_value = 7
        else:
            pawn_value = 1
            
        for i in range(1, 3):
            # break if top rank (avoid index error)
            if check_top_rank(color, checking_pos):
                # set points to max and break
                points = 3
                break;
            # check if next square has friendly pawn
            if bitboard[checking_pos] == pawn_value:
                return points
            else:
                if color == 'w':
                    checking_pos -= 8
                else:
                    checking_pos += 8
                    
                points = i

        return points

    # check king's file
    checking_pos = king_pos[0] - 8
    safety[0] += check_pawns_in_file('w', checking_pos)
    
    # check left and/or right file
    if king_edge[0] != 'L':
        checking_pos = king_pos[0] - 7
        safety[0] += check_pawns_in_file('w', checking_pos)
    if king_edge[0] != 'R':
        checking_pos = king_pos[0] - 9
        safety[0] += check_pawns_in_file('w', checking_pos)


    # ----- BLACK KING -----
    
    # check if king is on opposite top rank
    if check_top_rank('b', black_king):
        # max king safety points
        king_edge[1] = 'T'
    # check for position 0 (edge case to prevent division by 0)
    elif black_king == 0:
        king_edge[1] = 'L'
    # if king is at left edge
    elif black_king % 8 == 0:
        king_edge[1] = 'L'
    # if king is at right edge
    elif black_king % 8 == 7:
        king_edge[1] = 'R'


    # check king's file
    checking_pos = king_pos[1] + 8
    safety[1] += check_pawns_in_file('b', checking_pos)
    
    # check left and/or right file
    if king_edge[1] != 'L':
        checking_pos = king_pos[1] + 9
        safety[1] += check_pawns_in_file('b', checking_pos)
    if king_edge[1] != 'R':
        checking_pos = king_pos[1] + 7
        safety[1] += check_pawns_in_file('b', checking_pos)

    return safety


# Testing

# pgn = open("./data/test_games")
# first_game = chess.pgn.read_game(pgn)

# board_fen = get_random_pos(first_game)

# bitboard = convert_to_bitboard(board_fen)
    
# king_pos = find_kings(bitboard)
# king_safety = king_safety_eval(king_pos, "standard", bitboard)
# print(king_safety)

# board = chess.Board(board_fen)
# print(board)

# PAWN POINTS
# All 3 infront of king = 1
# Exponential will be + amount of squares pushed
# Standard will be + 1 for each square pushed
