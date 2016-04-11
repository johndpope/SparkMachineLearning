package com.lordjoe.machine_learning.tictactoe;

import org.testng.Assert;

import java.util.*;

import static com.lordjoe.machine_learning.tictactoe.TicTacToeBoard.getBoardSize;
import static com.lordjoe.machine_learning.tictactoe.Player.*;


/**
 * com.lordjoe.machine_learning.tictactoe.GamesRunner
 * User: Steve
 * Date: 4/6/2016
 */
public class GamesRunner {
    public static final double END_RATIO = getBoardSize() * getBoardSize() * 10;


    /**
     * pretty naive scores - counts only wins
     * @param b
     * @return
     */
    public static double scoreBoard(TicTacToeBoard b)   {
        Player winner = b.hasWin();
        if(winner != null)
            return winner.value;
         return 0;
      }


    private final Set<TicTacToeBoard> boards = new HashSet<>();
    private final Set<TicTacToeBoard> x_wins = new HashSet<>();
    private final Set<TicTacToeBoard> o_wins = new HashSet<>();
    private final Set<TicTacToeBoard> draw = new HashSet<>();
    private int runsSinceChange;
    private int totalGames;

    public GamesRunner() {
    }

    /**
     * build from a list of boards
     *
     * @param boardStrs
     */
    public GamesRunner(List<String> boardStrs) {
        this();
        for (String board : boardStrs) {
            boards.add(new TicTacToeBoard(board));
        }
        runsSinceChange = 0;
    }

    public double changeRatio() {
        if (boards.size() == 0) return 0;
        return runsSinceChange / (double) boards.size();
    }


    private void playGame() {
        TicTatToeGame g = new TicTatToeGame(new RandomPlayer(X), new RandomPlayer(O));
        while (!g.isEndState()) {
            addBoard(g.getBoard());
            g.makeMove();
        }
        addBoard(g.getBoard());
        totalGames++;
    }

    private void addBoard(TicTacToeBoard board) {
        if (boards.contains(board)) {  // not novel
            runsSinceChange++;
        } else {
            boards.add(board);    // novel
            runsSinceChange = 0;
        }
    }

    public List<String> getBoards() {
        List<String> holder = new ArrayList<String>();
        for (TicTacToeBoard board : boards) {
            holder.add(board.toString());
        }
        Collections.sort(holder);  // alphabetize

        return holder;
    }

    public void populate() {
        while (changeRatio() < END_RATIO) {
            playGame();
        }
        for (TicTacToeBoard board : boards) {
            Player p = board.hasWin();
            if (p == null) {
                if (board.isFull())
                    draw.add(board);
                else {
                    if (p == O)
                        o_wins.add(board);
                    else
                        x_wins.add(board);
                }
            }
        }
    }

    public static final double SCORE_FOR_WIN = 1;

    // evaluate the score for a board
    public double score(TicTacToeBoard test) {
        Player p = test.hasWin();
        if (p != null) {
            if (p == X)
                return SCORE_FOR_WIN;
            else
                return -SCORE_FOR_WIN;
        } else {
            return scoreFutures(test);
        }
    }

    private double scoreFutures(TicTacToeBoard test) {
        int numberXWins = countSubBoards(test, x_wins);
        int numberOWins = countSubBoards(test, o_wins);
        double xScore = SCORE_FOR_WIN * (numberXWins / (double)x_wins.size());
        double oScore = -SCORE_FOR_WIN * (numberOWins / (double)o_wins.size());
        return xScore + oScore;
    }

    private int countSubBoards(TicTacToeBoard test, Set<TicTacToeBoard> testSet) {
        int ret = 0;
        for (TicTacToeBoard board : testSet) {
            if (test.isSubBoard(board))
                ret++;
        }
        return ret;
    }

    private void showGameEvaluation() {
        TicTatToeGame g = new TicTatToeGame(new RandomPlayer(X), new RandomPlayer(O));
        while (!g.isEndState()) {
            TicTacToeBoard board = g.getBoard();
            double d = score(board);
            Player leading = null;
            if(d > 0.2)
                leading = X;
            if(d < -0.2)
                leading = O;

            if(leading != null)  {

            }
            addBoard(board);
            g.makeMove();
        }
        TicTacToeBoard board = g.getBoard();
        System.out.println();
        addBoard(board);
        totalGames++;
    }

    public static void main(String[] args) {
        GamesRunner runner = new GamesRunner();
        runner.populate();
        List<String> boards = runner.getBoards();
        Assert.assertEquals(5478, boards.size());


    }


}
