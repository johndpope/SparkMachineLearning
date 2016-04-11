package com.lordjoe.machine_learning.tictactoe;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.Serializable;

import static com.lordjoe.machine_learning.tictactoe.TicTacToeBoard.getTotalBoardSize;

/**
 * com.lordjoe.machine_learning.tictactoe.PlusMinusMapping
 * User: Steve
 * Date: 4/7/2016
 */
public class PlusMinusMapping implements IInputMapping,Serializable {

    /**
     * boardsize values -1 for O 1 for X 0 for empty
     * @param b  the board
     * @return
     */
    @Override
    public Vector boardToVector(TicTacToeBoard b) {
        double[] values = new double[getTotalBoardSize()];
        int index = 0;
        for (int i = 0; i < b.getBoardSize(); i++) {
            for (int j = 0; j < b.getBoardSize(); j++) {
                 Player p = b.getCell(i,j);
                if(p != null)
                    values[index] = p.value;
                index++;
            }
        }

        return Vectors.dense(values);
    }
}
