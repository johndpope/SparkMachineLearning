package com.lordjoe.machine_learning.tictactoe;

import java.util.List;

/**
 * com.lordjoe.machine_learning.tictactoe.Row
 * a line of positions on a given board
 * win happens when one player occupies all rows
 * User: Steve
 * Date: 4/7/2016
 */
public class Row {
    public final TicTacToeBoard board;
    public final Position[] cells = new Position[TicTacToeBoard.getBoardSize()];

    public Row(TicTacToeBoard board, List<Position> pcells) {
        this.board = board;
        for (int i = 0; i < pcells.size(); i++) {
            cells[i] = pcells.get(i);
          }
    }

    public Player getWinner()
    {
        Player test = board.getCell(cells[0]) ;
        if(test == null)
            return null;
        for (int i = 1; i < cells.length; i++) {
            if(test != board.getCell(cells[i]))
                return null;

        }
        return test;
    }
}
