import sys
import chess.pgn
import random
import copy
import re
import numpy as np
import pandas as pd


def addition_factorial(num):
    """
    Returns the addition factorial of a given number (e.g. 1+2+3 rather than 1*2*3)
    """
    
    return int(((num*num) + num) / 2)

def find_number_moves(moves):
    """
    Finds the number of moves in a given game

    :param moves: pgn of moves for a game
    :return: number of moves in the game as an integer
    """
    
    return int(re.findall(" (\d+)\.", moves)[-1])

def get_player_ratings(game):
    """
    Returns the ratings of each player

    :param game: The game to retrieve ratings from
    :return: Player ratings
    """
    
    return np.array([game.headers["WhiteElo"], game.headers["BlackElo"]])

def get_random_pos(game, move_number=-1):
    """
    Gets a random board position in a given game

    :param game: game to find position in
    :param move_number: move number to return position of
    :return: random board position as FEN
    """
    board = game.board()
    
    num_moves = find_number_moves(str(game.mainline_moves()))
    
    # Check move number is valid for given game
    if move_number > num_moves or move_number < -1:
        return "Move number out of range"
    
    # If move number is -1 then do random
    if(move_number == -1):
        rand = random.randrange(1, (2*num_moves))
        count = 0

        for move in game.mainline_moves():
            board.push(move)
            count += 1
            if count == rand:
                break
            
    else:
        rand = random.randrange(((2*move_number) - 2), ((2*move_number)))
        count = 0
        for move in game.mainline_moves():
            board.push(move)
            count += 1
            if count == rand:
                break
            
    next_move = board.pop()
            
    return [board.fen(), next_move]

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
    turn = 1

    # if white then set to white
    if 'w' in fen:
        turn = 0

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
    # check if given king position is in the opposite top rank
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

    :param king_pos: position of king to evaluate
    :param method: method to use (standard or exponential)
    :param bitboard: bitboard of position
    :return: value to represent evaluation of king safety
    
    """
    
    def check_king_edges(king_pos, color):
        """
        Checks if given king is at an edge of the board

        :param king_pos: position of king on bitboard
        :param color: color of king
        """
        
        # check if king is on opposite top rank
        if check_top_rank(color, king_pos):
            return 'T'
        # check for position 0 (edge case to prevent division by 0)
        elif king_pos == 0:
            return 'L'
        # if king is at left edge
        elif king_pos % 8 == 0:
            return 'L'
        # if king is at right edge
        elif king_pos % 8 == 7:
            return 'R'
        else:
            return 'N'
    
    def check_pawns_in_file(color, checking_pos):
        """
        Check given file for a friendly pawn

        :param color: color pawn to check for
        :param checking_pos: starting position to check for pawn (moves up the file from this position)
        """
        
        # initialize points
        points = 0
        
        # set value for respective pawn color
        if color == 'w':
            pawn_value = 7
        else:
            pawn_value = 1
        
        # loop 3 times (max safety rating is 3 for standard)
        for i in range(1, 4):
            # break if top rank (avoid index error)
            if check_top_rank(color, checking_pos):
                # set points to max and break
                if method == "standard":
                    points = 3
                else:
                    points = addition_factorial(3)
                break
            
            # check if next square has friendly pawn
            if bitboard[checking_pos] == pawn_value:
                return points
            else:
                # move up the file
                if color == 'w':
                    checking_pos -= 8
                else:
                    checking_pos += 8
                    
                # check method
                if method == "standard":
                    points = i
                else:
                    points = addition_factorial(i)

        return points


        
    # Determine pawn positions
    def check_files(color, king_edge, king_pos):
        """
        Checks appropriate files for pawns
        
        :param color: color pawns to check for
        """
        
        # initialize safety
        safety = 1
        
        # if color is white
        if color == 'w':
                 
            # check king's file
            checking_pos = king_pos[0] - 8
            
            # check method
            safety += check_pawns_in_file('w', checking_pos)
            
            # check left and/or right file
            if king_edge[0] != 'R':
                checking_pos = king_pos[0] - 7
                safety += check_pawns_in_file('w', checking_pos)
                
            if king_edge[0] != 'L':
                checking_pos = king_pos[0] - 9
                safety += check_pawns_in_file('w', checking_pos)
        else:
            
            # check king's file
            checking_pos = king_pos[1] + 8
            safety += check_pawns_in_file('b', checking_pos)
            
            # check left and/or right file
            if king_edge[1] != 'R':
                checking_pos = king_pos[1] + 9
                safety += check_pawns_in_file('b', checking_pos)
            if king_edge[1] != 'L':
                checking_pos = king_pos[1] + 7
                safety += check_pawns_in_file('b', checking_pos)
                
        return safety
                

    king_edge = [check_king_edges(king_pos[0], 'w'), check_king_edges(king_pos[1], 'b')]
    safety = [check_files('w', king_edge, king_pos), check_files('b', king_edge, king_pos)]

    return np.array(safety)

def central_control_eval(board_pos):
    """
    Takes a position (as a python-chess board) and returns values for white and black central control
    
    :param board_pos: python-chess board position
    """
    
    # default center control
    central_control = [0, 0]
    
    # 1 points for each piece in center
    # 1 point for each piece attacking center
    # center is e4, e5, d4, d5
        
    # center squares
    conversion = {
        chess.E4: "e4",
        chess.E5: "e5",
        chess.D4: "d4",
        chess.D5: "d5",
    }
    
    center_attacked = [chess.E4, chess.E5, chess.D4, chess.D5]
        
    for square in center_attacked:
        
        # check attacked squares
        if(board_pos.is_attacked_by(chess.WHITE, square)):
            central_control[0] += 1
        if(board_pos.is_attacked_by(chess.BLACK, square)):
            central_control[1] += 1
            
        # check occupied squares
        if(str(board_pos.piece_at(chess.parse_square(conversion[square]))).isupper()):
            central_control[0] += 1
        elif(str(board_pos.piece_at(chess.parse_square(conversion[square]))).islower()):
            central_control[1] += 1
        
    return np.array(central_control)

def encode_move(move, method="std"):
    if method == "hash":
        return hash(move)
    else:
        mapping_dict = {
            'a': '1',
            'b': '2',
            'c': '3',
            'd': '4',
            'e': '5',
            'f': '6',
            'g': '7',
            'h': '8',
            'r': '1',
            'n': '2',
            'b': '3',
            'q': '4',
        }
        
        move = [*str(move)]
        for index, char in enumerate(move):
            if char.isalpha():
                move[index] = mapping_dict[char]
            
        move = ''.join(move)
        return int(move)
        
def test(game):
    for move in game.mainline_moves():
        print(encode_move(move), "\n")

def create_model_input(game, method, testing_move_number=-1):
    """
    Takes the game and creates an input for a machine learning model
    
    :param game: Game to create input from
    :param method: method for king safety evaluation (standard or exponential)
    :param testing_move_number: used for choosing a specific position instead of random (-1 is random pos)
    """
    
    # get board position and next move
    temp = get_random_pos(game, testing_move_number)
    board_pos = temp[0]
    next_move = temp[1]

    # convert to bitboard
    bitboard = convert_to_bitboard(board_pos)

    # get king safety and central control values
    king_pos = find_kings(bitboard)
    king_safety = king_safety_eval(king_pos, method, bitboard)
    central_control = central_control_eval(chess.Board(board_pos))

    # get player ratings
    player_ratings = get_player_ratings(game)
    
    # get turn color
    turn = find_turn(board_pos)
    
    # encode next move
    next_move = encode_move(next_move, "std")
    
    # change bitboard from list to string
    # bitboard = "".join([str(i) for i in bitboard])

    data = [bitboard, king_safety[0], king_safety[1], central_control[0], central_control[1], player_ratings[0], player_ratings[1], turn, next_move]
    
    return data

def generate_df():
    """
    Main function that runs the preprocessing on the chess games database
    """
    
    # access games database8
    pgn = open("../data/test_games")
    
    # create list of inputs
    inputs = [create_model_input(chess.pgn.read_game(pgn), "exp") for i in range(0, 3)]
    
    columns = ["bitboard", "w_safety", "b_safety", "w_central", "b_central", "w_rating", "b_rating", "turn", "next_move"]
    
    # convert to dataframe
    df = pd.DataFrame(inputs, columns=columns)
    
    # Convert bitboard to its own columns for input into model
    for i in range(64):
        df[f'piece_{i}'] = df['bitboard'].apply(lambda x: x[i])
        
    # drop bitboard columns
    df.drop(columns=['bitboard'], inplace=True)
    
    return df
    
if __name__ == "__main__":
    generate_df()
