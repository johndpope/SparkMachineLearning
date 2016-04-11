package com.lordjoe.machine_learning.tictactoe;

import org.junit.Assert;
import org.testng.annotations.Test;

import static com.lordjoe.machine_learning.tictactoe.Player.O;
import static com.lordjoe.machine_learning.tictactoe.Player.X;

/**
 * com.lordjoe.machine_learning.tictactoe.GameTests
 * User: Steve
 * Date: 4/6/2016
 */
public class GameTests {

    @Test
    public void testGame() {
        TicTatToeGame g = new TicTatToeGame(new RandomPlayer(X), new RandomPlayer(O));
        int movesLeft = TicTacToeBoard.getTotalBoardSize();
        TicTacToeBoard lastBoard = g.getBoard();
        int size = lastBoard.getLegalMoves().size();
        Assert.assertEquals(movesLeft--, size);
        while (!g.isEndState()) {
            g.makeMove();
            TicTacToeBoard board = g.getBoard();
            // we could get to board by adding move to lastBoard
            Assert.assertTrue(lastBoard.isSubBoard(board));
            lastBoard = board;
            int size1 = board.getLegalMoves().size();
            Assert.assertEquals(movesLeft--, size1);
        }
        TicTacToeBoard board = g.getBoard();
        Assert.assertTrue(board.isFull() || g.getWinner() != null);

        TicTacToeBoard b1 = new TicTacToeBoard(board.toString());
        Assert.assertEquals(b1, board);

        b1 = new TicTacToeBoard(board);
        Assert.assertEquals(b1, board);
    }

    @Test
    public void testGames() {
        for (int i = 0; i < 500; i++) {
            testGame();

        }
    }

}
