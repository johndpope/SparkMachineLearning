package com.lordjoe.machine_learning.tictactoe;

import com.lordjoe.machine_learning.minamax.IMinaMaxMove;

import javax.annotation.Nonnull;
import java.util.*;

import static com.lordjoe.machine_learning.tictactoe.Player.O;
import static com.lordjoe.machine_learning.tictactoe.Player.X;

/**
 * com.lordjoe.machine_learning.tictactoe.RandomPlayer
 * this player choses a move randomly from legal moves
 * nothing special for TicTacToe
 * User: Steve
 * Date: 4/6/2016
 */
public class MinaMaxPlayer implements IStrategy {
    public static final Random RND = new Random();

    public final Player player;



    public MinaMaxPlayer(Player player) {
        this.player = player;
    }

    @Nonnull
    @Override
    public Player getPlayer() {
        return player;
    }

    @Override
    public TicTacToeMove choseMove(@Nonnull TicTatToeGame g) {
        double factor = player.value;
        if(g.isEndState())
            throw new IllegalArgumentException("only chose moves in open games");
        List<TicTacToeMove>  best = new ArrayList<>();
        double bestScore = -1000;
        List<? extends IMinaMaxMove> nextMoves = g.getNextMoves();
        for (IMinaMaxMove nextMove : nextMoves) {
            TicTatToeGame aftermove = (TicTatToeGame)g.makeTrialMove(nextMove);
            double score = factor * aftermove.getMinamaxScore();
           // System.out.println("score " + score + "/n" + nextMove + "\n" + aftermove.toPrettyPrintString());
            if(score > bestScore)   {
                best.clear();
                best.add((TicTacToeMove)nextMove);
                  bestScore = score;
            }
            else {
                if(score == bestScore)   {
                    best.add((TicTacToeMove)nextMove);
                }
            }

        }
        if(best.size() > 1) {
             return best.get(RND.nextInt(best.size())); // make a random choice
        }
        else {
            return best.get(0);
        }
    }

    public static void main(String[] args) {
        int matchGames = 1000;
        MinaMaxPlayer x = new MinaMaxPlayer(X);
        MinaMaxPlayer o = new MinaMaxPlayer(O);

        RandomPlayer rpx = new RandomPlayer(X);
        RandomPlayer rpo = new RandomPlayer(O);

        Match match;
        String play;

        match = new Match(matchGames, rpx, rpo);
        play = match.play();
        System.out.println( "random x and o " + play);

        match = new Match(matchGames, rpx, o);
        play = match.play();
        System.out.println( "random vs o " + play);

        match = new Match(matchGames, x, o);
        play = match.play();
        System.out.println( " " + play);



    }

}
