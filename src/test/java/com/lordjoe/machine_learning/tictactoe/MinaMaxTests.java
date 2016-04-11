package com.lordjoe.machine_learning.tictactoe;

import com.lordjoe.machine_learning.minamax.IMinaMaxMove;
import org.junit.Assert;
import org.junit.Test;

import static com.lordjoe.machine_learning.tictactoe.Player.O;
import static com.lordjoe.machine_learning.tictactoe.Player.X;

/**
 * com.lordjoe.machine_learning.tictactoe.MinaMaxTests
 * User: Steve
 * Date: 4/8/2016
 */
public class MinaMaxTests {

    /**
     * minamax should always play to a draw
     */
    @Test
    public void minamaxPlaysToDraws() {
        int matchGames = 1000;
        MinaMaxPlayer x = new MinaMaxPlayer(X);
        MinaMaxPlayer o = new MinaMaxPlayer(O);
        Match match;
        String play;


        match = new Match(matchGames, x, o);
        play = match.play();
        int draws = match.getDraws();
        Assert.assertEquals(matchGames, draws);

    }

    /**
     * O  X
     * O
     * X
     */

    public static final String X_WIN_IN_2 = "O X O X  ";

    @Test
    public void xWinIn2() {
        MinaMaxPlayer x = new MinaMaxPlayer(X);
        MinaMaxPlayer o = new MinaMaxPlayer(O);
        TicTatToeGame game = new TicTatToeGame(x, o, X_WIN_IN_2);
        game.setMover(X);
        TicTacToeBoard board = game.getBoard();

        String actual = board.toString();
        Assert.assertEquals(X_WIN_IN_2, actual.replace("_", " ").replace("|", ""));     // board is what I think

        // properly score o wins
        IMinaMaxMove oWins = new TicTacToeMove(O, new Position(2, 2));
        game.setMover(O);
        TicTatToeGame aftermove = (TicTatToeGame) game.makeTrialMove(oWins);
        double score = aftermove.getMinamaxScore();
        Assert.assertEquals(-1, score, 0.001); // o wins in 1

        IMinaMaxMove badMove = new TicTacToeMove(X, new Position(1, 0));
        game.setMover(X);
        aftermove = (TicTatToeGame) game.makeTrialMove(badMove);
        score = aftermove.getMinamaxScore();
        Assert.assertEquals(-0.9, score, 0.001); // o wins in 1

        IMinaMaxMove goodMove = new TicTacToeMove(X, new Position(2, 2));
        game.setMover(X);
        aftermove = (TicTatToeGame) game.makeTrialMove(goodMove);
        score = aftermove.getMinamaxScore();
        Assert.assertEquals(0.81, score, 0.001); // x wins in 2

        IMinaMaxMove badMove2 = new TicTacToeMove(X, new Position(1, 2));
        game.setMover(X);
        aftermove = (TicTatToeGame) game.makeTrialMove(badMove);
        score = aftermove.getMinamaxScore();
        Assert.assertEquals(-0.9, score, 0.001); // o wins in 1


        TicTacToeMove move = x.choseMove(game);
        Assert.assertEquals(new Position(2, 2), move.cell);

    }

    /**
     * X  O
     * X
     * O
     */
    public static final String O_WIN_IN_2 = "X O X O  ";

    @Test
    public void oWinIn2() {
        MinaMaxPlayer x = new MinaMaxPlayer(X);
        MinaMaxPlayer o = new MinaMaxPlayer(O);
        TicTatToeGame game = new TicTatToeGame(x, o, X_WIN_IN_2);
        game.setMover(O);
        TicTacToeMove move = o.choseMove(game);
        Assert.assertEquals(new Position(2, 2), move.cell);

    }
}
