# ---------------- IMPORTS ---------------- #

import sys
import chess.pgn
import random
import copy
import time
import re
import os
import numpy as np
import pandas as pd
import argparse
from alive_progress import alive_bar

# ---------------- GLOBALS ---------------- #

MODEL_PREFIX = '../models/'
DATA_PREFIX = '../data/'
PIECE_VALUES = {
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

def addition_factorial(num):
    """
    Returns the addition factorial of a given number (e.g. 1+2+3 rather than 1*2*3)
    
    :param num: number to calculate with
    
    :return: addition factorial number
    """
    
    return int(((num*num) + num) / 2)

def find_number_moves(moves):
    """
    Finds the number of moves in a given game

    :param moves: pgn of moves for a game
    
    :return: number of moves in the game as an integer
    """
    
    # check if move set is null
    if moves == "" or moves == None:
        return 0
    
    # return number of moves
    return int(re.findall(r'(\d+)\.', moves)[-1])

def get_player_ratings(game):
    """
    Returns the ratings of each player

    :param game: the game to retrieve ratings from
    
    :return: player ratings
    """
    
    # return player ratings
    return np.array([game.headers["WhiteElo"], game.headers["BlackElo"]])

def get_random_pos(game, move_number=-1, turn=1):
    """
    Gets a random board position in a given game

    :param game: game to find position in
    :param move_number: move number to return position of
    :return: [random board position as FEN, next move]
    """
    
    # convert board into boar object
    board = game.board()
    
    # find the total number of moves
    num_moves = find_number_moves(str(game.mainline_moves()))
    if num_moves <= 0:
        return None
    
    # Check move number is valid for given game
    if move_number > num_moves or move_number < -1:
        return "Move number out of range"
    
    # If move number is -1 then do random
    if(move_number == -1):
        rand = random.randrange(1, (2*num_moves))
        count = 0

        # push each move until we hit random number
        for move in game.mainline_moves():
            board.push(move)
            count += 1
            if count == rand:
                break
            
    else:
        # if black or white
        if turn == 'b':
            count = 0
        else:
            count = 1

        # push each move until we hit move number
        for move in game.mainline_moves():
            board.push(move)
            count += 1
            if count == (move_number*2):
                break
    
    # pop the last move off of the board
    next_move = board.pop()
    
    # return board position and next move
    return [board.fen(), next_move]

def convert_to_bitboard(fen):
    """
    Converts a board position to bitboard representation

    :param fen: fen of given position to convert
    
    :return: bitboard array of position
    """

    # convert fen to board
    board = str(chess.Board(fen))
    
    def convert_to_value(x):
        """
        Converts a character to its respective integer value
    
        :param x: character to convert
        
        :return: integer values respetive to character
        """
        
        # convert
        return PIECE_VALUES[x]

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

    # return as array
    return np.array([w_king_pos[0][0], b_king_pos[0][0]])

def check_top_rank(color, pos):
    """
    Checks if given king position is in the opposite top rank
    
    :param color: color of king to check for
    :param pos: position of king
    
    : return: True if king position is in the opposite top rank
    """

    # check for color
    if color == 'w':
        # if king position is in top rank
        if pos <= 7:
            return True
        else:
            return False
    elif color == 'b':
        # if king position is in top rank
        if pos >= 57:
            return True
        else:
            return False

def check_king_edges(king_pos, color):
        """
        Checks if given king is at an edge of the board

        :param king_pos: position of king on bitboard
        :param color: color of king
        
        :return: T, L, R or N -> Top, Left, Right or None
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
    
def check_pawns_in_file(color, checking_pos, bitboard, method):
    """
    Check given file for a friendly pawn

    :param color: color pawn to check for
    :param checking_pos: starting position to check for pawn (moves up the file from this position)
    
    :return: calculated points value
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
   
def check_files(color, king_edge, king_pos, bitboard, method):
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
        safety += check_pawns_in_file('w', checking_pos, bitboard, method)
        
        # check left and/or right file
        if king_edge[0] != 'R':
            checking_pos = king_pos[0] - 7
            safety += check_pawns_in_file('w', checking_pos, bitboard, method)
            
        if king_edge[0] != 'L':
            checking_pos = king_pos[0] - 9
            safety += check_pawns_in_file('w', checking_pos, bitboard, method)
    else:
        
        # check king's file
        checking_pos = king_pos[1] + 8
        safety += check_pawns_in_file('b', checking_pos, bitboard, method)
        
        # check left and/or right file
        if king_edge[1] != 'R':
            checking_pos = king_pos[1] + 9
            safety += check_pawns_in_file('b', checking_pos, bitboard, method)
        if king_edge[1] != 'L':
            checking_pos = king_pos[1] + 7
            safety += check_pawns_in_file('b', checking_pos, bitboard, method)
            
    return safety

def king_safety_eval(king_pos, method, bitboard):
    """
    Evaluates the safety of given king (cap of 3 pushes for each pawn)

    :param king_pos: position of king to evaluate
    :param method: method to use (standard or exponential)
    :param bitboard: bitboard of position
    :return: value to represent evaluation of king safety
    
    """
                
    king_edge = [check_king_edges(king_pos[0], 'w'), check_king_edges(king_pos[1], 'b')]
    safety = [check_files('w', king_edge, king_pos, bitboard, method), check_files('b', king_edge, king_pos, bitboard, method)]

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

def encode_moves_binary(moves):
    """
    Encodes a position to binary columns (to be used with binary and vector encodings)
    
    :param moves: list of moves in format E.g. "e5d5" or for promotion to queen "e4e8q"
    """
    
    # data to create dataframe from
    data = []
    
    # loop through every row of dataframe
    for row in moves:
        
        # get the start and end square
        start_square, end_square = row[:2], row[2:]
        
        # get start and end columns and ranks
        start_column, start_rank = start_square[0], start_square[1]
        end_column, end_rank = end_square[0], end_square[1]
        
        # blank row entry
        blank_list = [0] * 36
        
        # set needed start and end positions indexes to 1
        blank_list[ord(start_column) - 88] = 1
        blank_list[int(start_rank)] = 1
        blank_list[ord(end_column) - 72] = 1
        blank_list[int(end_rank) + 16] = 1
        
        # if there is a promotion
        if len(row) > 4:
            promotion_index = 35
            # determine the correct column based on piece
            match (row[4]):
                case 'q':
                    promotion_index = 32
                    break
                case 'r':
                    promotion_index = 33
                    break
                case 'b':
                    promotion_index = 34
                    break
                case 'n':
                    promotion_index = 35
                    break
                
            # set correct promotion index to 1
            blank_list[promotion_index] = 1    
        
        # append list to data
        data.append(blank_list)
        
        
    # create dataframe from data
    encoded_df = pd.DataFrame(data, columns=['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 'sa', 'sb', 'sc', 'sd', 'se',
                                             'sf', 'sg', 'sh', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'ea', 'eb',
                                             'ec', 'ed', 'ee', 'ef', 'eg', 'eh', 'promote_q', 'promote_r', 'promote_b',
                                             'promote_n'])
    
    return encoded_df
    
def encode_moves_binary_vector(moves):
    """
    Encodes the "next_move" with binary start position and vector move
    
    :param moves: list of moves in format E.g. "e4d5" or for promotion to queen "e4e8q"
    """
        
    # data to create dataframe from
    data = []
    
    # loop through every row of dataframe
    for row in moves:
        
        # get the start and end square
        start_square, end_square = row[:2], row[2:]
        
        # get start and end columns and ranks
        start_column, start_rank = start_square[0], start_square[1]
        end_column, end_rank = end_square[0], end_square[1]
                
        # calculate columns and ranks moved
        columns_moved = ord(end_column) - ord(start_column)
        ranks_moved = ord(end_rank) - ord(start_rank)
        
        # blank row entry
        blank_list = [0] * 22
        
        # set needed start position indexes to 1
        blank_list[ord(start_column) - 89] = 1
        blank_list[int(start_rank) - 1] = 1
        
        # columns and ranks moved
        blank_list[16] = columns_moved
        blank_list[17] = ranks_moved
        
        
        # if there is a promotion
        if len(row) > 4:
            promotion_index = 21
        # determine the correct column based on piece
            match (row[4]):
                case 'q':
                    promotion_index = 18
                    break
                case 'r':
                    promotion_index = 19
                    break
                case 'b':
                    promotion_index = 20
                    break
                case 'n':
                    promotion_index = 21
                    break
                
            # set correct promotion index to 1
            blank_list[promotion_index] = 1    
        
        # append list to data
        data.append(blank_list)
        
        
    # create dataframe from data
    encoded_df = pd.DataFrame(data, columns=['1', '2', '3', '4', '5', '6', '7', '8', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                                             'columns_moved', 'ranks_moved', 'promote_q', 'promote_r', 'promote_b', 'promote_n'])
    
    return encoded_df

def encode_move_std(move):    
    # Convert to list
    move = [*str(move)]
    
    # Check if promotion in move
    if len(move) > 4:
        move[4] = move[4].upper()
                
    # Map of move letter to number
    mapping_dict = {
            'a': '1',
            'b': '2',
            'c': '3',
            'd': '4',
            'e': '5',
            'f': '6',
            'g': '7',
            'h': '8',
            'R': '1',
            'N': '2',
            'B': '3',
            'Q': '4',
        }
        
    # Loop through each char, if not number then convert
    for index, char in enumerate(move):
        if char.isalpha():
            move[index] = mapping_dict[char]
    
    # Convert back to string and return
    move = ''.join(move)
    return str(move)

def get_legal_moves(position):
    return [position.san(move) for move in position.legal_moves]

def test(game):
    for move in game.mainline_moves():
        print(encode_move_std(move), "\n")

def create_model_input(game, r_to, r_from, k_safety_method, testing_move_number=-1, turn=1):
    """
    Takes the game and creates an input for a machine learning model
    
    :param game: Game to create input from
    :param method: method for king safety evaluation (standard or exponential)
    :param testing_move_number: used for choosing a specific position instead of random (-1 is random pos)
    """    
    
    # check game is not none
    if game is None:
        return -1
    
    if not re.search(r"600[^\d]*", game.headers["TimeControl"]):
        return None
    
    # check elo is in range
    if r_to != 0:
        w_elo = int(game.headers["WhiteElo"])
        b_elo = int(game.headers["BlackElo"])
        if w_elo < r_from or w_elo > r_to or b_elo < r_from or b_elo > r_to:
            return None
    
    # get board position and next move
    temp = get_random_pos(game, testing_move_number, turn)
    
    if temp == None:
        return None
    
    board_pos = temp[0]
    next_move = str(temp[1])
    if len(next_move) == 5: return None
    
    # convert to bitboard
    bitboard = convert_to_bitboard(board_pos)

    # get king safety and central control values
    king_pos = find_kings(bitboard)
    king_safety = king_safety_eval(king_pos, k_safety_method, bitboard)
    central_control = central_control_eval(chess.Board(board_pos))

    # get player ratings
    player_ratings = get_player_ratings(game)
    
    # get turn color
    turn = find_turn(board_pos)
    
    # change bitboard from list to string
    # bitboard = "".join([str(i) for i in bitboard])

    data = [board_pos, bitboard, king_safety[0], king_safety[1], central_control[0], central_control[1], player_ratings[0], player_ratings[1], turn, next_move]
    
    return data

def generate_df(filename, num_inputs, r_from, r_to, start_index=0, k_safety_method='std', encode_method='std'):
    """
    Main function that runs the preprocessing on the chess games database
    
    :param dbpath: path to the database of chess games
    :param k_safety_method: method for evaluating king safety (standard or exponential)
    :param encode-method: method for encoding next move (value, binary, binary + vector)
    """
    
    # set the file path
    dbpath = f'../data/{filename}'
    
    # access games database
    pgn = open(dbpath)
    
    
    # get to start index position
    if start_index > 0:
        index_count = 0
        print("\nSkipping to start index...")
        with alive_bar(start_index, bar="classic2", stats=False, spinner=None) as skip_bar:
            while index_count < start_index:
                chess.pgn.skip_game(pgn)
                index_count += 1
                skip_bar()

    
    if num_inputs == -1:
        # create list of inputs
        inputs = []
        
        game = chess.pgn.read_game(pgn)
        while (game != None):
            singleInput = create_model_input(game, k_safety_method)
            if singleInput != None and singleInput != -2:
                inputs.append(singleInput)
            game = chess.pgn.read_game(pgn)
    else:
        # create list of inputs
        inputs = []
        
        # set count for number of inputs
        count = 0
        print("\nPreprocessing data...")
        with alive_bar(num_inputs, bar="classic2", stats=False, spinner=None) as bar:
            while count < num_inputs:
                singleInput = create_model_input(chess.pgn.read_game(pgn), r_to, r_from, k_safety_method)
                if singleInput == -1:
                    print(f'\nNo more games in file, created {count} inputs. Exiting.')
                    break
                if singleInput != None:
                    inputs.append(singleInput)
                    count += 1
                    bar()
    
    columns = ["board_pos", "bitboard", "w_safety", "b_safety", "w_central", "b_central", "w_rating", "b_rating", "turn", "next_move"]
    
    # convert to dataframe    
    df = pd.DataFrame(inputs, columns=columns)
    
    # Convert bitboard to its own columns for input into model
    for i in range(64):
        df[f'square_{i}'] = df['bitboard'].apply(lambda x: x[i])
        
    
    # encode next_move column
    if encode_method == "vector":
        encoded_df = encode_moves_binary_vector(df['next_move'].to_numpy())
        df = pd.concat([df, encoded_df], axis=1)
    elif encode_method == "binary":
        encoded_df = encode_moves_binary(df['next_move'].to_numpy())
        df = pd.concat([df, encoded_df], axis=1)
    else:
        # encode and convert to start and end squares
        df['next_move_encoded'] = df['next_move'].apply(encode_move_std)
        df['start_square'] = df['next_move_encoded'].str[:2]
        df['end_square'] = df['next_move_encoded'].str[2:]
        df = df.drop(columns=['next_move_encoded'])
        
    # drop bitboard column
    df.drop(columns=['bitboard'], inplace=True)
    
    # drop next_move column
    df.drop(columns=['next_move'], inplace=True)
    
    # remove extenion from filename is needed
    if filename.endswith('.pgn'):
        filename = filename[:-4]
    
    # save to csv file
    df.to_csv(DATA_PREFIX + f'{filename}.csv', index=False)

def create_single_input(filename, move_number, turn=1, k_safety_method='std'):
    # set the file path
    dbpath = f'../data/{filename}'
    
    # access games database
    pgn = open(dbpath)
    
    singleInput = create_model_input(chess.pgn.read_game(pgn), k_safety_method, move_number, turn)
    
    columns = ["board_pos", "bitboard", "w_safety", "b_safety", "w_central", "b_central", "w_rating", "b_rating", "turn", "next_move"]
    
    # convert to dataframe    
    df = pd.DataFrame([singleInput])
    df.columns = columns
    
    # Convert bitboard to its own columns for input into model
    for i in range(64):
        df[f'square_{i}'] = df['bitboard'].apply(lambda x: x[i])
    
    # drop columns
    df = df.drop(columns=['next_move', 'bitboard'])
    
    # remove extenion from filename is needed
    if filename.endswith('.pgn'):
        filename = filename[:-4]
        
    # save to csv file
    df.to_csv(DATA_PREFIX + f'{filename}_single.csv', index=False)
    
def main():
    # ARGUMENT HANDLING
    parser = argparse.ArgumentParser(description="Model Selection")
    parser.add_argument('type', choices=['single', 'multiple'], help="Type of function to run")
    parser.add_argument('--n_inputs', type=int, required=False, help="[Multiple Inputs] Number of inputs to use (-1 for all)")
    parser.add_argument('--start', type=int, required=False, default=0, help="[Multiple Inputs] Index of game to start at (default = 0)")
    parser.add_argument('--r_from', type=int, required=False, default=0, help="[Multiple Inputs] Miniumu elo rating of either player")
    parser.add_argument('--r_to', type=int, required=False, default=0, help="[Multiple Inputs] Maximum elo rating of either player")
    parser.add_argument('--move', type=int, required=False, help="[Single Input] Specify move number for single prediction")
    parser.add_argument('--turn', choices=['w', 'b'], required=False, help="[Single Input] Turn (White or Black)")
    parser.add_argument('--file', type=str, required=True, help="Name of file to process (including any extensions)")
    args = parser.parse_args()
    
    # check if given filename exists
    if not os.path.isfile(f'../data/{args.file}'):
        print("ERROR: File not found")
        exit()
    elif args.type == 'single':
        if args.move is None:
            print("Please specify a move number (-m)")
            exit()
        elif args.turn is None:
            create_single_input(args.file, args.move)
        else:
            create_single_input(args.file, args.move, args.turn)
            
    elif args.type == 'multiple':
        if args.n_inputs is None:
            print("Please specify the number of inputs to create (-n)")
            exit()
        elif (args.r_to != 0 and args.r_from == 0) or (args.r_to == 0 and args.r_from != 0):
            print("If either r_to or r_from is used, both must be specified.")
            exit()
        else:
            generate_df(args.file, args.n_inputs, args.r_from, args.r_to, args.start)
            
    
    
    

if __name__ == "__main__":
    main()