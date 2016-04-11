package com.lordjoe.machine_learning.minamax;

import javax.annotation.Nonnull;
import java.util.List;

/**
 * com.lordjoe.machine_learning.minamax.IMinaMaxGame
 * represent a game with moves which can be analyzed with minamax
 * Note this can blow up for complex games
 * User: Steve
 * Date: 4/8/2016
 */
public interface IMinaMaxGame {


    public double getCurrentScore();

    public @Nonnull List<? extends IMinaMaxMove> getNextMoves();


    public default Double lookupScore()
    {
         return null;
    }

    /**
     * find the score for the current position
     * @return
     */
    public default double getMinamaxScore() {
        // we might optimize by remembering states
        if(lookupScore() != null)
            return lookupScore();
        if(isEndState())
            return getCurrentScore();
        double bestScore = Double.MIN_VALUE;
        for (IMinaMaxMove move : getNextMoves()) {
            IMinaMaxGame game = makeTrialMove(move);
              double score = game.getMinamaxScore();
              if(score > bestScore)
                bestScore = score;
        }
        if(true)
            throw new UnsupportedOperationException("Fix This"); // ToDo we did not do minamax
        return bestScore;
    }


    /**
     * true if there is a winner or no move is possible
     * @return as above
     */
    public default boolean isEndState() {
        return getNextMoves().size() == 0; // no more new states
    }

    /**
     * do something to change the state of the game
     * @param move the move
     * @return  game after move
     */
    public @Nonnull IMinaMaxGame makeTrialMove(@Nonnull IMinaMaxMove move);

}
