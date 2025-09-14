import numpy as np

def is_winner(board, player):
    for i in range(3):
        if all(board[i, j] == player for j in range(3)):
            return True
        if all(board[j, i] == player for j in range(3)):
            return True
    if all(board[i, i] == player for i in range(3)):
        return True
    if all(board[i, 2-i] == player for i in range(3)):
        return True
    return False

def is_full(board):
    return not any(board[i, j] == '' for i in range(3) for j in range(3))

def get_available_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == '']

def minimax(board, player, maximizing):
    if is_winner(board, 'X'): return 1, None
    if is_winner(board, 'O'): return -1, None
    if is_full(board): return 0, None
    moves = get_available_moves(board)
    if maximizing:
        best_score, best_move = -float('inf'), None
        for i, j in moves:
            board[i, j] = 'X'
            score, _ = minimax(board, 'O', False)
        