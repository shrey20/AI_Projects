
import random
import time
import numpy as np
import copy

""" An object representation for an AI game player for the game Teeko2."""
class Teeko2Player:
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']
    
    """ Initializes a Teeko2Player object by randomly selecting red or black as its
        piece color."""
    def __init__(self):
        
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    
        
    
    """ Selects a (row, col) space for the next move. Given the state of the board, generates all possible successor states and chooses
        the next state based of minimax algorithm. If the game is in drop phase(all peices aren't on the board), drops a new peice else 
        moves an existing peice according to the rules of the game."""
    def make_move(self, state):
        move = []
        drop_phase = True   
        count_b = 0
        count_r = 0
        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] == 'b':
                    count_b += 1
                elif state[i][j] == 'r':
                    count_r += 1
                else:
                    continue
        if count_b == 4 and count_r == 4:
            drop_phase = False
        
        if drop_phase:
              # Generate all successive states
              successors = self.succ(state)
              # Select move based on minimax algorithm
              for eachstate in successors:
                    stepval = self.min_val(float("-inf"), float("inf"), eachstate[0], 2)
                    max_value = -np.inf
                    if max_value < stepval:
                        max_value = stepval
                        next_state = eachstate
              move = next_state[1]
                
        

        if not drop_phase:
              successors = self.succ(state)
              for eachstate in successors:
                stepval = self.min_val(float("-inf"), float("inf"), eachstate[0], 0)
                max_value = -np.inf
                if max_value < stepval:
                    max_value = stepval
                    next_state = eachstate
              move = next_state[1]
        return move
    
    """Given the state of the board returns a list of list of possible moves. """
    def succ(self, state):
        possible = []
        
        
        drop_phase = True   
        count_b = 0
        count_r = 0
        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] == 'b':
                    count_b += 1
                elif state[i][j] == 'r':
                    count_r += 1
                else:
                    continue
        if count_b == 4 and count_r == 4:
            drop_phase = False
            
        if drop_phase == True:
            for i in range(len(state)):
                for j in range(len(state[i])):
                    succ = copy.deepcopy(state)
                    if succ[i][j] == ' ':
                        succ[i][j] = self.my_piece
                        mov = [(i,j)]
                        possible.append([succ, mov])
                    
                    else:
                        continue
        else:
            for i in range(len(state)):
                for j in range(len(state[i])):
                    succ = copy.deepcopy(state)
                    if succ[i][j] == self.my_piece:
                        for x in range(i-1, i +1):
                            for y in range(j-1, j+1):
                                if (x in range(0,4) and y in range(0,4)):
                                    
                                    if succ[x][y] == ' ':
                                        succ[x][y] = self.my_piece
                                        succ[i][j] = ' '
                                        movement = [(x,y), (i,j)]
                                        possible.append([succ, movement])
                                    else:
                                        continue
        return possible
                        
        
    """ Validates the opponent's next move against the internal board representation."""
    def opponent_move(self, move):
        
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    """ Modifies the board representation using the specified move and piece."""
    def place_piece(self, move, piece):
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    """ Formatted printing for the board """
    def print_board(self):
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    """ Generates a heuristic to quantify 'how close' a certain state is to the winning state. 
    It addes points for every peice that is connected (according to the game rules and reduces points
    for every opponent peice that is connected, which results in a heuristic that not only tries to find the minimum number of 
    moves required to connect the pieces and win but also simultaneously tries to stop the opponect from connecting their pieces. """
    def heuristic_game_value(self, state):
        x = self.game_value(state) #'how close' player is to winning
        y = 0.0 #'how close' opponent is to winning.
        if (x == 1 or x == -1):
            return x
        else:
            x = 0
            for i in range(4):
                for j in range(4):
                    if state[i][j] == ' ':
                        continue
                    else:
                        # Evaluates vertical connections
                        if state[i][j] == state[i+1][j]:
                            if i+2 <= 4 and state[i][j] == state[i+2][j]:
                                if state[i][j] == self.my_piece:
                                    x = max(0.75, y)
                                else:
                                    y = min(-0.75, y)
                            else:
                                if state[i][j] == self.my_piece:
                                    x = max(0.5, x)
                                else:
                                    y =  min(-0.5, y)
                            
                        
                        # Evaluates horizontal connections
                        if state[i][j] == state[i][j+1]:
                            if j+2 <= 4 and state[i][j] == state[i][j+2]:
                                if state[i][j] == self.my_piece:
                                    x = max(0.75, x)
                                else:
                                    y = min(-0.75, y)
                            else:
                                if state[i][j] == self.my_piece:
                                    x =  max(0.5, x)
                                else:
                                    y = min(-0.5, y) 
                
                # Evaluates \ diagonal connections
                if state[i][i] == state[i+1][i+1]:
                    if i+2 <= 4 and state[i][i] == state[i+2][i+2]:
                        if state[i][i] == self.my_piece:
                            x = max(0.75, x)
                        else:
                            y = min(-0.75, y)
                    else:
                        if state[i][i] == self.my_piece:
                            x = max(0.5, x)
                        else:
                            y = min(-0.5, y)
                            
                # Evaluates / diagonal connections
                if state[i][4-i] == state[i+1][3 - i]:
                    if state[i][4-i] == state[i+2][2 - i]:
                        if state[i][i] == self.my_piece:
                            x = max(0.75, x)
                        else:
                            y = min(-0.75, y)
                    else:
                        if state[i][i] == self.my_piece:
                            x = max(0.75, x)
                        else:
                            y = min(-0.5, y)
                
                # Checks square winning condition.
                for i in range(3):
                    for j in range(3):
                        if state[i][j] != ' ':
                            a = self.my_piece
                            b = self.opp
                            sqr = (state[i+2][j+2], state[i+2][j], state[i][j+2])
                            h = sqr.count(a)
                            k = sqr.count(b)
                            if (h == 1):
                                x = max(0.5, x)
                            if (h == 2):
                                x = max(0.75, y)
                            if (k == 1):
                                y = min(-0.5, y)
                            if (k ==2):
                                y = min(-0.75, y)
                
                return x+y 
    
    """ Max value that a player can achieve considering the opponent tries to minimize the  heuristic value."""
    def max_val(self, alpha,beta,state, depth):
        x = self.game_value(state)
        if (x == 1 or x == -1):
            return x
        if depth == 0:
            return self.heuristic_game_value(state)
        else:
            successors = self.succ(state)
            for s in successors:
                alpha = max(alpha, self.min_val(alpha, beta, s[0], depth +1 ))
                # pruning uisng beta value
                if alpha >= beta: 
                    return beta
            return alpha
        
    """ Min vlaue that a player can restrict the opponent to considering the opponent tries to maximize the nueristic value."""    
    def min_val(self, alpha,beta,state, depth):
        x = self.game_value(state)
        if (x == 1 or x == -1):
            return x
        if depth == 0:
            return self.heuristic_game_value(state)
        else:
            successors = self.succ(state)
            for s in successors:
                alpha = min(alpha, self.max_val(alpha, beta, s[0], depth +1 ))
                #pruning using alpha value
                if alpha <= beta:
                    return beta
            return alpha
                    
        
     """ Checks the current board status for a win condition."""
    def game_value(self, state):
        # check horizontal wins
        for row in state:
            #print(row)
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check \ diagonal wins
        for i in range(2):
            for j in range(2):
                if state[i][j] != ' ' and state[i][j] == state[i+1][j + 1] == state[i+2][j+2] == state[i+3][j+3]:
                    return 1 if state[i][i] == self.my_piece else -1
        # TODO: check / diagonal wins
        if state[3][0] != ' ' and state[3][0] == state[2][1] == state[1][2] == state[0][3]:
                    return 1 if state[3][0] == self.my_piece else -1
        
        if state[0][4] != ' ' and state[0][4] == state[1][3] == state[2][2] == state[3][1]:
                    return 1 if state[0][4] == self.my_piece else -1
        
        if state[1][3] != ' ' and state[1][3] == state[2][2] == state[3][1] == state[4][0]:
                    return 1 if state[1][3] == self.my_piece else -1
        
        if state[1][4] != ' ' and state[1][4] == state[2][3] == state[3][2] == state[4][1]:
                    return 1 if state[1][4] == self.my_piece else -1
        
        # TODO: check 3x3 square corners wins
        for i in range(3):
            for j in range(3):
                if state[i][j] != ' ' and state[i][j] == state[i+2][j] ==state[i][j+2] == state[i+2][j+2] and state[i+1][j+1] == ' ':
                    return 1 if state[i][j] == self.my_piece else -1
                    
        return 0 # no winner yet

#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY 
#
def main():
    print('Hello, this is Samaritan')
    ai = Teeko2Player()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2
        
    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()

