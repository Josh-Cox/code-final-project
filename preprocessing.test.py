from src.preprocessing import *
import unittest

class TestPreprocessing(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        """
        Sets up variables once for use in all tests
        """
        self.test_data = open("unit_testing/testing.pgn")
        self.test_game = chess.pgn.read_game(self.test_data)
        self.test_moves = str(self.test_game.mainline_moves())
        self.test_fen = self.test_game.board().fen()
        self.test_move = 'e4d7'
        self.test_move_encoded = '5447'
        
        # Different test positions as bitboards
        self.test_bitboard  = np.array([2, 3, 4, 5, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 10, 11, 12, 10, 9, 8])
        self.test_opposite_rank_white = convert_to_bitboard('rnb1Kb1r/ppp2ppp/3p4/4p1Q1/2qNP1n1/3P4/PPP2kPP/RNB2B1R w - - 0 1')
        self.test_opposite_rank_black = convert_to_bitboard('rnb2b1r/ppp2ppp/3pK3/4p1Q1/2qNP1n1/3P4/PPP3PP/RNB1kB1R w - - 0 1')
        self.test_good_king_safey = convert_to_bitboard('rnb1kb1r/pppppppp/8/6Q1/2qN2n1/3P4/PPP1PPPP/RNB1KB1R w - - 0 1')
        self.test_bad_king_safey = convert_to_bitboard('rnb1kb1r/ppp3pp/3p4/4ppQ1/2qNP1n1/3P4/PPP2P1P/RNB1KB1R w - - 0 1    ')
        self.test_left_edge_king_safety = convert_to_bitboard('1nb2b1k/ppp5/3p2pp/4ppQ1/2qNP1n1/1P1P4/P1P2P1P/KNB2B2 w - - 0 1')
        self.test_right_edge_king_safety = convert_to_bitboard('knb2b1r/ppp3pp/3p4/4ppQ1/2qNP1n1/3P4/PPP2P1P/RNB2B1K w - - 0 1')
        
        # Different positions to test central control
        self.test_central_control_one = chess.Board('knb2b1r/ppp3pp/3p4/4ppQ1/2qNP1n1/3P4/PPP2P1P/RNB2B1K w - - 0 1')
        
        # Test the machine learning model input
        self.test_model_input_bitboard = convert_to_bitboard(self.test_fen)
        self.test_model_input_king_safety = np.array([4, 1])
        self.test_model_input_central_control = np.array([4, 5])
        self.test_elo = ['1961', '1618']

    def test_find_number_moves(self):
        """
        Tests function to find the number of moves in a given set of moves
        """
        
        self.assertEqual(find_number_moves(self.test_moves), 30, "Number of moves is wrong")
        
    def test_get_player_ratings(self):
        """
        Tests function to get the ratings of each player in a given game
        """
        
        np.array_equal(get_player_ratings(self.test_game), self.test_elo)
        
    def test_get_random_position(self):
        """
        Tests function to pick a random position from a given game
        """
        
        # Get random position
        test_rand_pos = get_random_pos(self.test_game)[0]
        
        # Check this is correct FEN format by trying to create board object
        try:
            chess.Board(test_rand_pos)
        except:
            raise ValueError("Random position given not valid FEN")
            
    def test_convert_to_bitboard(self):
        """
        Tests function that takes a FEN and returns a bitboard
        """
        
        np.testing.assert_allclose(convert_to_bitboard(self.test_fen), self.test_bitboard, rtol=0, atol=0, err_msg="Bitboard is not correct")
        
    def test_find_turn(self):
        """
        Tests function that takes a fen and returns who's current turn it is (which color)
        """
        self.assertEqual(find_turn(self.test_fen), 0)
        
    def test_find_kings(self):
        """
        Tests function that takes a bitboard and returns the indexes of the kings
        """
        
        correct_pos = np.array([60, 4])
        np.testing.assert_allclose(find_kings(self.test_bitboard), correct_pos, rtol=0, atol=0, err_msg="King positions are not correct")
        
        correct_pos = np.array([20, 60])
        np.testing.assert_allclose(find_kings(self.test_opposite_rank_black), correct_pos, rtol=0, atol=0, err_msg="King positions are not correct")
        
        correct_pos = np.array([4, 53])
        np.testing.assert_allclose(find_kings(self.test_opposite_rank_white), correct_pos, rtol=0, atol=0, err_msg="King positions are not correct")
         
    def test_check_top_rank(self):
        """
        Tests function that takes the color and king position and returns whether it is on the opposite rank
        """
        self.assertTrue(check_top_rank('w', 4), "White king did not flag as opposite rank")
        self.assertTrue(check_top_rank('b', 60), "Black king did not flag as opposite rank")
        self.assertFalse(check_top_rank('w', 60), "White king flagged as opposite rank")
        self.assertFalse(check_top_rank('b', 4), "Black king flagged as opposite rank")
        
    def test_king_safety_eval(self):
        """
        Tests function that takes king positions, method and bitboard, returns a king safety evaluation
        """
        
        # Checking standard
        np.testing.assert_allclose(king_safety_eval(np.array([60, 4]), self.test_good_king_safey), np.array([2, 1]), rtol=0, atol=0, err_msg="King safety evaluation incorrect")
        np.testing.assert_allclose(king_safety_eval(np.array([60, 4]), self.test_bad_king_safey), np.array([4, 6]), rtol=0, atol=0, err_msg="King safety evaluation incorrect")
        np.testing.assert_allclose(king_safety_eval(np.array([63, 0]), self.test_right_edge_king_safety), np.array([4, 1]), rtol=0, atol=0, err_msg="King safety evaluation incorrect")
        np.testing.assert_allclose(king_safety_eval(np.array([56, 7]), self.test_left_edge_king_safety), np.array([2, 3]), rtol=0, atol=0, err_msg="King safety evaluation incorrect")
      
    def test_central_control_eval(self):
        """
        Tests function that takes a board position and returns a value for central control for each color
        """
        np.testing.assert_allclose(central_control_eval(self.test_central_control_one), np.array([4, 5]), rtol=0, atol=0, err_msg="Central control is incorrect")
        
    def test_encode_move_std(self):
        """
        Tests function to encode a given move
        """
        
        self.assertEqual(encode_move_std(self.test_move), self.test_move_encoded)
        
    def test_create_model_inputs(self):
        """
        Tests function that returns an input for a machine learning model
        """
    
        # test input for the function
        test_input = create_model_input(self.test_game, testing_move_number=20)

        # Check correct FEN format
        fen = test_input[0]
        try:
            chess.Board(fen)
        except:
            raise ValueError("Random position given not valid FEN")
                
        # two correct bitboards (white or black's turn)
        correct_bitboard = np.array([0, 0, 0, 2, 0, 0, 0, 0, 1, 3, 1, 1, 6, 1, 0, 0, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 
                                       0, 0, 0, 0, 1, 0, 0, 0, 0, 7, 0, 0, 0, 0, 9, 0, 7, 0, 0, 0, 0, 0, 7, 7, 11, 10, 
                                       0, 7, 7, 0, 8, 0, 0, 0, 12, 0, 0, 0])
                        
        # check bitboards        
        np.array_equal(test_input[1], correct_bitboard)
        
        
        # check king safety values
        np.array_equal(test_input[2], 6)
        np.array_equal(test_input[3], 7) 
        
        # check central control
        self.assertEqual(test_input[4], 4)
        self.assertEqual(test_input[5], 1)
        
        # check ratings      
        self.assertEqual(test_input[6], self.test_elo[0])
        self.assertEqual(test_input[7], self.test_elo[1])
        
        # check turn
        self.assertTrue(test_input[8] == 0 or test_input[8] == 1)
        
        # check next move
        move = test_input[9]
        regex1 = r'[a-e]'
        regex2 = r'[1-8]'
        self.assertRegex(move[0], regex1)
        self.assertRegex(move[2], regex1)
        self.assertRegex(move[1], regex2)
        self.assertRegex(move[3], regex2)
                        
if __name__ == '__main__':
        unittest.main()