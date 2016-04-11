package com.lordjoe.machine_learning.tictactoe;

import com.lordjoe.machine_learning.minamax.IMinaMaxMove;

/**
 * com.lordjoe.machine_learning.tictactoe.TicTacToeMove
 * one movee for one player
 * User: Steve
 * Date: 4/8/2016
 */
public class TicTacToeMove implements IMinaMaxMove{

    public final Player player;
    public final Position cell;

    public TicTacToeMove(Player player, Position cell) {
        this.player = player;
        this.cell = cell;
    }

    @Override
    public String toString() {
        return  player.toString() + " "  + cell.toString();
    }
}
