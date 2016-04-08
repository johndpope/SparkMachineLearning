package com.lordjoe.machine_learning.tictactoe;

import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.io.Serializable;

/**
 * com.lordjoe.machine_learning.tictactoe.IInputMapping
 * User: Steve
 * Date: 4/7/2016
 */
public interface IInputMapping  extends Serializable {

    /**
     * how does the board map to features
     * @param b  the board
     * @return
     */
    public Vector  boardToVector(Board b);

    /**
     * given the feature mapping map to a LabeledPoint
     * @param b  the board
     * @param label  label
     * @return
     */
    public default LabeledPoint boardToPoint(Board b, double label)  {
        Vector  v = boardToVector(b);
        return new LabeledPoint(label,v);
    }

}
