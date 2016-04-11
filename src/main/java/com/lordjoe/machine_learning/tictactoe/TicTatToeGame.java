package com.lordjoe.machine_learning.tictactoe;


import com.lordjoe.machine_learning.minamax.IMinaMaxGame;
import com.lordjoe.machine_learning.minamax.IMinaMaxMove;
import org.apache.commons.collections.map.HashedMap;

import javax.annotation.Nonnull;
import java.util.*;

import static com.lordjoe.machine_learning.tictactoe.Player.O;
import static com.lordjoe.machine_learning.tictactoe.Player.X;

/**
 * com.lordjoe.machine_learning.tictactoe.Game
 * a game between two players - X always moves first (X and O are arbitrary and
 * really mean forst and second player
 * User: Steve
 * Date: 4/6/2016
 */
public class TicTatToeGame implements IMinaMaxGame {
    // remember scoring of particular boards
    private static Map<String, Double> cachedScores = new HashedMap();

    private TicTacToeBoard board = new TicTacToeBoard();
    private final Map<Player, IStrategy> players = new HashMap<Player, IStrategy>();
    private Player mover; // X always starts
    private Player winner;
    private TicTatToeGame baseGame; // mimamaxs replicates games - this is where we startes

    /**
     * build a game with strategies
     *
     * @param x strategy for x player
     * @param o strategy for o player
     */
    public TicTatToeGame(IStrategy x, IStrategy o) {
        if (x.getPlayer() == o.getPlayer()) throw new IllegalArgumentException("players must be different");
        players.put(x.getPlayer(), x);
        players.put(o.getPlayer(), o);
        mover = x.getPlayer();
        baseGame = this;
    }

    /**
     * constructor which can set board
     *
     * @param x        strategy for x player
     * @param o        strategy for o player
     * @param boordStr string like "XO XOO   "
     */
    public TicTatToeGame(IStrategy x, IStrategy o, String boordStr) {
        this(x, o, new TicTacToeBoard(boordStr));
    }


    /**
     * constructor which can set board
     *
     * @param x     strategy for x player
     * @param o     strategy for o player
     * @param boord board to set
     */

    public TicTatToeGame(IStrategy x, IStrategy o, TicTacToeBoard pBoard) {
        this(x, o);
        board = new TicTacToeBoard(pBoard);
    }

    /**
     * copy constructor
     *
     * @param copy
     */
    public TicTatToeGame(TicTatToeGame copy) {
        this(copy.getStrategy(copy.mover), copy.getStrategy(copy.mover.next()));
        board = copy.getBoard();
        baseGame = copy.baseGame;
    }

    public IStrategy getStrategy(Player p) {
        return players.get(p);

    }

    public void setMover(Player mover) {
        this.mover = mover;
    }

    /**
     * true if there is a winner or the board is full
     *
     * @return
     */
    @Override
    public boolean isEndState() {
        if (winner != null)
            return true;
        if (board.isFull())   // separate case for debugging
            return true;
        return false;
    }

    /**
     * return the winner - first found or null if no winner
     * makeTrialMove will set winner
     *
     * @return
     */
    public Player getWinner() {
        return winner;
    }

    /**
     * make a single move - this resets mover and
     * may set winner
     *
     * @return true if there is a winner or the board is full
     */
    public boolean makeMove() {
        TicTacToeMove position = players.get(mover).choseMove(this);
        makeMove(position);
        if (isEndState()) {
            mover = null;
            return true; // game over;
        }
        return false; // still on
    }


    public void makeMove(TicTacToeMove move) {
        if (winner != null)
            throw new IllegalStateException("no moves allowed if game has a winner");
        if (mover != move.player)
            throw new IllegalStateException("wrong player " + move.player);

        board.makeMove(move);
        winner = board.hasWin();
        mover = mover.next();
    }

    /**
     * X plays then Y until finished
     *
     * @return
     */
    public Player playGame() {
        while (!isEndState()) {
            makeMove();
        }
        return winner;
    }

    /**
     * record all moves from first until winner or draw
     *
     * @return
     */
    public List<TicTacToeBoard> getPlayBoards() {
        List<TicTacToeBoard> holder = new ArrayList<TicTacToeBoard>();
        while (!isEndState()) {
            makeMove();
            holder.add(getBoard());
        }

        return holder;
    }


    public TicTacToeBoard getBoard() {
        return new TicTacToeBoard(board);
    }


    @Override
    public double getCurrentScore() {
        Player winner = getWinner();
        if (winner != null)
            return winner.value;
        return 0;
    }

    /**
     * return all moves possible for the current mover
     *
     * @return
     */
    @Override
    public List<? extends IMinaMaxMove> getNextMoves() {
        List<TicTacToeMove> holder = new ArrayList<TicTacToeMove>();
        for (Position cell : board.getLegalMoves()) {
            holder.add(new TicTacToeMove(mover, cell));
        }
        return holder;
    }

    @Override
    public Double lookupScore() {
        return lookupScore(mover);
    }

    public Double lookupScore(Player p) {
        String key = board.toString();
        return baseGame.cachedScores.get(p.toString() + "|" + key);
    }

    public void cacheScore(TicTacToeBoard board, double score) {
        String b = mover.toString() + "|" + board.toString();

        Map<String, Double> cachedScores = baseGame.cachedScores;
        if (cachedScores.containsKey(b)) {
            double oldscore = cachedScores.get(b);
            if (oldscore == score)
                return;
            cachedScores.put(b, score);
        } else {
            cachedScores.put(b, score);
        }
    }

    public static void breakHere() {

    }

    public static final double MOVE_PENALTY = 0.9;
    public static final double TOO_LOW = -1000;

    /**
     * ok just like  getRawMinamaxScore but only on first level so I can work out issues
     *
     * @return
     */
    @Override
    public double getMinamaxScore() {
        return getMinamaxScore(0);
    }

    /**
     * ok just like  getRawMinamaxScore but only on first level so I can work out issues
     *
     * @return
     */

    protected double getMinamaxScore(int level) {
        double winningScore = mover.value;
        // see if we have already computed the score
        Double cachedScore = lookupScore();
        double realCache = 0;
        if (cachedScore != null) {
            realCache = cachedScore;
            return realCache;
        }
        // see if the board is full or there is a winning position
        if (isEndState()) {
            double currentScore = getCurrentScore();
            cacheScore(getBoard(), currentScore);
            return currentScore;
        }
        // maximize if X is the player minimize for O
        double bestScore = (mover == X) ? TOO_LOW : -TOO_LOW;

        List<? extends IMinaMaxMove> nextMoves = getNextMoves();
        IMinaMaxMove bestMove = null;  // will be the best move
        for (IMinaMaxMove move : nextMoves) {
            TicTatToeGame game = (TicTatToeGame) makeTrialMove(move);
            //System.out.println(move.toString() + "\n" + game.toString() + " level " + level);
            double rawScore = game.getMinamaxScore(level + 1);
            double score = rawScore * MOVE_PENALTY;

            if (rawScore == winningScore) {
                bestScore = score;
                bestMove = move;
                break; // win in 1 means stop searching
            }

            switch (mover) {
                case X:
                    if (score > bestScore) {
                        bestScore = score;
                        bestMove = move;
                    }
                    break;
                case O:    // chose minimum for O move
                    if (score < bestScore) {
                        bestScore = score;
                        bestMove = move;
                    }
                    break;


            }
        }
        cacheScore(getBoard(), bestScore);
//        if (cachedScore != null) {
//            if (realCache !=  bestScore) {
//                System.out.println(realCache + " != " + bestScore);
//            }
//        }
        return bestScore;   // take out x-o factor give strength for X
    }


    /**
     * ok just like  getRawMinamaxScore but only on first level so I can work out issues
     *
     * @return
     */

    protected double getMinamaxScoreX(int level) {
        if (mover == O)
            breakHere();
        // we are evaluating a move so after the move it is the other players turn
        int factor = mover.value;
        double winningScore = factor;
        Double cachedScore = lookupScore();
        double realCache = 0;
        if (cachedScore != null) {
            realCache = cachedScore * factor;
            return realCache;
        }
        if (isEndState()) {
            double currentScore = getCurrentScore();
            cacheScore(getBoard(), currentScore);
            return currentScore;
        }
        double bestScore = TOO_LOW;
        List<? extends IMinaMaxMove> nextMoves = getNextMoves();
        IMinaMaxMove bestMove = null;
        for (IMinaMaxMove move : nextMoves) {
            TicTatToeGame game = (TicTatToeGame) makeTrialMove(move);
            // System.out.println(move.toString() + "\n" + game.toString() + " level " + level);
            double rawScore = game.getMinamaxScore(level + 1);

            double score = factor * rawScore * MOVE_PENALTY;
            if (rawScore == winningScore) {
                bestScore = score;
                bestMove = move;
                break; // win in 1 means stop searching
            }
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }
        cacheScore(getBoard(), bestScore * factor);
//        if (cachedScore != null) {
//            if (realCache !=  bestScore) {
//                System.out.println(realCache + " != " + bestScore);
//            }
//        }
        return factor * bestScore;   // take out x-o factor give strength for X
    }


    /**
     * return the game state if move made
     * as a copy of the original game
     *
     * @param move the move
     * @return
     */
    @Override
    public IMinaMaxGame makeTrialMove(@Nonnull IMinaMaxMove move) {
        TicTatToeGame ret = new TicTatToeGame(this);
        ret.makeMove((TicTacToeMove) move);
        return ret;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        if (isEndState()) {
            if (winner != null)
                sb.append("winner " + winner + " ");
            else
                sb.append("draw ");
        } else {
            sb.append("move " + mover + " ");
        }
        sb.append(board.toString());
        return sb.toString();
    }

    public String toPrettyPrintString() {
        StringBuilder sb = new StringBuilder();
        if (isEndState()) {
            if (winner != null)
                sb.append("winner " + winner + " ");
            else
                sb.append("draw ");
        } else {
            sb.append("move " + mover + " ");
        }
        sb.append("\n");
        sb.append(board.toPrettyPrintString());
        return sb.toString();
    }
}
