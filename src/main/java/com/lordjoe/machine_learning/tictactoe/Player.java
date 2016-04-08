package com.lordjoe.machine_learning.tictactoe;

/**
 * com.lordjoe.machine_learning.tictactoe.player
 * X always moves first (X and O are arbitrary and
 *  really mean forst and second player)
 * User: Steve
 * Date: 4/6/2016
 */
public enum Player {
    O(-1),X(1);
    public final int value;

    Player(int value) {
        this.value = value;
    }

    public Player next()
    {
        switch (this)    {
            case X: return O;
            case O: return X;
         }
        throw new IllegalStateException("never get here");
    }
}
