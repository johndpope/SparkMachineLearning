package com.lordjoe.machine_learning.tictactoe;

import javax.annotation.Nonnull;
import java.io.Serializable;

/**
 * com.lordjoe.machine_learning.tictactoe.IPlayer
 * essentially a strategy for chosing moces
 * User: Steve
 * Date: 4/6/2016
 */
public interface IStrategy extends Serializable {

    /**
     * playing X or O
     * @return
     */
    public @Nonnull Player getPlayer();

    /**
     * giver a board chose a move
     * @param b
     * @return
     */
    public TicTacToeMove choseMove(@Nonnull TicTatToeGame b);

}
