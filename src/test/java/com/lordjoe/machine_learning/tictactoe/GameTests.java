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
        Game g = new Game(new RandomPlayer(X), new RandomPlayer(O));
        int movesLeft = Board.getTotalBoardSize();
        Board lastBoard = g.getBoard();
        int size = lastBoard.getLegalMoves().size();
        Assert.assertEquals(movesLeft--, size);
        while (!g.isFinished()) {
            g.makeMove();
            Board board = g.getBoard();
            // we could get to board by adding move to lastBoard
            Assert.assertTrue(lastBoard.isSubBoard(board));
            lastBoard = board;
            int size1 = board.getLegalMoves().size();
            Assert.assertEquals(movesLeft--, size1);
        }
        Board board = g.getBoard();
        Assert.assertTrue(board.isFull() || g.getWinner() != null);

        Board b1 = new Board(board.toString());
        Assert.assertEquals(b1, board);

        b1 = new Board(board);
        Assert.assertEquals(b1, board);
    }

    @Test
    public void testGames() {
        for (int i = 0; i < 500; i++) {
            testGame();

        }
    }

}
