from src.preprocessing import *
import unittest

class TestPreprocessing(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        """
        Sets up variables once for use in all tests
        """
        self.test_data = open("data/testing.pgn")
        self.test_game = chess.pgn.read_game(self.test_data)
        self.test_moves = str(self.test_game.mainline_moves())
        self.test_fen = self.test_game.board().fen()
        
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
        self.test_white_elo = 1961
        self.test_black_elo = 1618

    def test_find_number_moves(self):
        """
        Tests function to find the number of moves in a given set of moves
        """
        self.assertEqual(find_number_moves(self.test_moves), 31, "Number of moves is wrong")
        
    def test_get_random_position(self):
        """
        Tests function to pick a random position from a given game
        """
        
        # Get random position
        test_rand_pos = get_random_pos(self.test_game)[0]
        
        regex = r"^((([pnbrqkPNBRQK1-8]{1,8})\/?){8})\s+(b|w)\s+(-|K?Q?k?q)\s+(-|[a-h][3-6])\s+(\d+)\s+(\d+)\s*$"

        # Check this is correct FEN format
        self.assertRegex(test_rand_pos, regex)
    
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
        np.testing.assert_allclose(king_safety_eval(np.array([60, 4]), "standard", self.test_good_king_safey), np.array([2, 1]), rtol=0, atol=0, err_msg="King safety evaluation incorrect")
        np.testing.assert_allclose(king_safety_eval(np.array([60, 4]), "standard", self.test_bad_king_safey), np.array([4, 6]), rtol=0, atol=0, err_msg="King safety evaluation incorrect")
        np.testing.assert_allclose(king_safety_eval(np.array([63, 0]), "standard", self.test_right_edge_king_safety), np.array([4, 1]), rtol=0, atol=0, err_msg="King safety evaluation incorrect")
        np.testing.assert_allclose(king_safety_eval(np.array([56, 7]), "standard", self.test_left_edge_king_safety), np.array([2, 3]), rtol=0, atol=0, err_msg="King safety evaluation incorrect")
        
        # Checking Exponential
        np.testing.assert_allclose(king_safety_eval(np.array([60, 4]), "exponential", self.test_good_king_safey), np.array([2, 1]), rtol=0, atol=0, err_msg="King safety evaluation incorrect")
        np.testing.assert_allclose(king_safety_eval(np.array([60, 4]), "exponential", self.test_bad_king_safey), np.array([5, 8]), rtol=0, atol=0, err_msg="King safety evaluation incorrect")
        np.testing.assert_allclose(king_safety_eval(np.array([63, 0]), "exponential", self.test_right_edge_king_safety), np.array([7, 1]), rtol=0, atol=0, err_msg="King safety evaluation incorrect")
        np.testing.assert_allclose(king_safety_eval(np.array([56, 7]), "exponential", self.test_left_edge_king_safety), np.array([2, 3]), rtol=0, atol=0, err_msg="King safety evaluation incorrect")

        
    def test_central_control_eval(self):
        """
        Tests function that takes a board position and returns a value for central control for each color
        """
        np.testing.assert_allclose(central_control_eval(self.test_central_control_one), np.array([4, 5]), rtol=0, atol=0, err_msg="Central control is incorrect")
        
        
    def test_create_model_inputs(self):
        """
        Tests function that returns an input for a machine learning model
        """
    
        # test input for the function
        print(create_model_input(self.test_game, "standard", 20))
        test_input = create_model_input(self.test_game, "standard", 20)
        
        regex = r"^((([pnbrqkPNBRQK1-8]{1,8})\/?){8})\s+(b|w)\s+(-|K?Q?k?q)\s+(-|[a-h][3-6])\s+(\d+)\s+(\d+)\s*$"

        # Check correct FEN format
        self.assertRegex(test_input[0], regex)
                
        # two correct bitboards (white or black's turn)
        correct_bitboards = [np.array([ 2,  0,  0,  0,  0,  0,  6,  0,  1,  0,  0,  2,  0,  1,  1,  1,  0,
        0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  9,  0,  0,  0,  0,  0,  0,
        0,  0,  7,  0,  0,  0,  0,  7,  0,  0,  0,  9,  0,  0,  7,  0,  0,
        0,  0,  7,  7, 10,  0,  0,  0,  8,  0,  0, 12,  0]),
        np.array([ 2,  0,  0,  2,  0,  0,  6,  0,  1,  0,  0,  0,  0,  1,  1,  1,  0,
        0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  9,  0,  0,  0,  0,  0,  0,
        0,  0,  7,  0,  0,  0,  0,  7,  0,  0,  0,  9,  0,  0,  7,  0,  0,
        0,  0,  7,  7, 10,  0,  0,  0,  8,  0,  0, 12,  0])]
                        
        # check bitboards        
        self.assertTrue(
            np.array_equal(test_input[1], correct_bitboards[0]) or
            np.array_equal(test_input[1], correct_bitboards[1]))
        
        
        # check king safety
        self.assertTrue((np.array_equal(test_input[2], 4)) or (np.array_equal(test_input[2], 6)) and
            (np.array_equal(test_input[3], 1) or np.array_equal(test_input[3], 7))
                        
        ) 
        
        # check central control
        self.assertEqual(test_input[4], 5)
        self.assertEqual(test_input[5], 1)
        
        # check ratings      
        self.assertEqual(test_input[6], '1486')
        self.assertEqual(test_input[7], '1417')
        
        # check turn
        self.assertTrue(test_input[8] == 0 or test_input[8] == 1)
        
                

if __name__ == '__main__':
        unittest.main()