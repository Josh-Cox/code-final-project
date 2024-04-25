# ---------------- IMPORTS ---------------- #

import chess.pgn
import random
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
    
def check_pawns_in_file(color, checking_pos, bitboard):
    """
    Check given file for a friendly pawn

    :param color: color pawn to check for
    :param checking_pos: starting position to check for pawn (moves up the file from this position)
    :param bitboard: board postion as a bitboard
    
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
            points = 3
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
            points = i


    return points
   
def check_files(color, king_edge, king_pos, bitboard):
    """
    Checks appropriate files for pawns
    
    :param color: color pawns to check for
    :param king_edge: if king is at a board edge - which one
    :param bitboard: board position as a bitboard

    :return: safety points
    """
    
    # initialize safety
    safety = 1
    
    # if color is white
    if color == 'w':
                
        # check king's file
        checking_pos = king_pos[0] - 8
        
        # check method
        safety += check_pawns_in_file('w', checking_pos, bitboard)
        
        # check left and/or right file
        if king_edge[0] != 'R':
            checking_pos = king_pos[0] - 7
            safety += check_pawns_in_file('w', checking_pos, bitboard)
            
        if king_edge[0] != 'L':
            checking_pos = king_pos[0] - 9
            safety += check_pawns_in_file('w', checking_pos, bitboard)
    else:
        
        # check king's file
        checking_pos = king_pos[1] + 8
        safety += check_pawns_in_file('b', checking_pos, bitboard)
        
        # check left and/or right file
        if king_edge[1] != 'R':
            checking_pos = king_pos[1] + 9
            safety += check_pawns_in_file('b', checking_pos, bitboard)
        if king_edge[1] != 'L':
            checking_pos = king_pos[1] + 7
            safety += check_pawns_in_file('b', checking_pos, bitboard)
            
    return safety

def king_safety_eval(king_pos, bitboard):
    """
    Evaluates the safety of given king (cap of 3 pushes for each pawn)

    :param king_pos: position of king to evaluate
    :param bitboard: bitboard of position

    :return: value to represent evaluation of king safety
    """
                
    king_edge = [check_king_edges(king_pos[0], 'w'), check_king_edges(king_pos[1], 'b')]
    safety = [check_files('w', king_edge, king_pos, bitboard), check_files('b', king_edge, king_pos, bitboard)]

    return np.array(safety)

def central_control_eval(board_pos):
    """
    Takes a position (as a python-chess board) and returns values for white and black central control
    
    :param board_pos: python-chess board position

    :return: central control value
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
    
    # define center squares
    center_attacked = [chess.E4, chess.E5, chess.D4, chess.D5]
        
    # loop through squares
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

def encode_move_std(move):
    """
    Encode a move from letters and number to only numbers

    :param move: move to encode

    :return: encoded move
    """ 

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
    """
    Get all of the legal move in the position

    :param position: board position

    :return: list of legal moves
    """

    return [position.san(move) for move in position.legal_moves]

def create_model_input(game, r_to=0, r_from=0, testing_move_number=-1, turn=1, time_control_check=True):
    """
    Takes the game and creates an input for a machine learning model
    
    :param game: game to create input from
    :param r_from: lower bound for player rating
    :param r_to: upper bound for player rating
    :param testing_move_number: used for choosing a specific position instead of random (-1 is random pos)
    :param turn: turn to pick (w or b)
    :parm time_control_check: bool to filter only 10 minute games
    """    
    
    # check game is not none
    if game is None:
        return -1
    
    # check game is ten minutes
    if time_control_check and not re.search(r"600[^\d]*", game.headers["TimeControl"]):
        return None
    
    # check elo is in range
    r_to = int(r_to)
    r_from = int(r_from)
    if r_to != 0:
        w_elo = int(game.headers["WhiteElo"])
        b_elo = int(game.headers["BlackElo"])
        if w_elo < r_from or w_elo > r_to or b_elo < r_from or b_elo > r_to:
            return None
    
    # get board position and next move
    temp = get_random_pos(game, testing_move_number, turn)
    
    # if no positon returned
    if temp == None:
        return None
    
    # filter out promotional moves
    board_pos = temp[0]
    next_move = str(temp[1])
    if len(next_move) == 5: return None
    
    # convert to bitboard
    bitboard = convert_to_bitboard(board_pos)

    # get king safety and central control values
    king_pos = find_kings(bitboard)
    king_safety = king_safety_eval(king_pos, bitboard)
    central_control = central_control_eval(chess.Board(board_pos))

    # get player ratings
    player_ratings = get_player_ratings(game)
    
    # get turn color
    turn = find_turn(board_pos)
    
    # return all of the data
    data = [board_pos, bitboard, king_safety[0], king_safety[1], central_control[0], central_control[1], player_ratings[0], player_ratings[1], turn, next_move]
    
    return data

def generate_df(filename, num_inputs, r_from, r_to, start_index=0):
    """
    Main function that runs the preprocessing on the chess games database
    
    :param filename: path to the database of chess games
    :param num_inputs: number of inputs to generate from file (if larger than file then whole file will be used)
    :param r_from: lower bound for player rating
    :param r_to: upper bound for player rating
    :param start_index: index of game to start from

    :return: save pandas dataframe of preprocessed inputs to file
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
            singleInput = create_model_input(game)
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
                singleInput = create_model_input(chess.pgn.read_game(pgn), r_to, r_from)
                if singleInput == -1:
                    print(f'\nNo more games in file, created {count} inputs. Exiting.')
                    break
                if singleInput != None:
                    inputs.append(singleInput)
                    count += 1
                    bar()
    
    # generate column names
    columns = ["board_pos", "bitboard", "w_safety", "b_safety", "w_central", "b_central", "w_rating", "b_rating", "turn", "next_move"]
    
    # convert to dataframe    
    df = pd.DataFrame(inputs, columns=columns)
    
    # Convert bitboard to its own columns for input into model
    for i in range(64):
        df[f'square_{i}'] = df['bitboard'].apply(lambda x: x[i])
        
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

def create_single_input(filename, move_number, turn=1):
    """
    Preprocesses a single game input

    :param filename: path of file of pgn game to use
    :param move_number: move number to predict
    :param turn: turn to predict

    :return: save pandas dataframe of input to file
    """

    # set the file path
    dbpath = f'../data/{filename}'
    
    # access games database
    pgn = open(dbpath)
    singleInput = create_model_input(game=chess.pgn.read_game(pgn), testing_move_number=move_number, turn=turn, time_control_check=False)
    columns = ["board_pos", "bitboard", "w_safety", "b_safety", "w_central", "b_central", "w_rating", "b_rating", "turn", "next_move"]
    
    # convert to dataframe    
    df = pd.DataFrame(data=[singleInput], columns=columns)
    
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
    # ---------------- ARGUMENT HANDLING ---------------- #

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
            create_single_input(filename=args.file, move_number=args.move, turn=args.turn)
            
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