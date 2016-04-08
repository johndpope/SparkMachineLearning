package com.lordjoe.machine_learning;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.io.Serializable;

/**
 * com.lordjoe.machine_learning.examples.StringToLabeledPoint
 * User: Steve
 * Date: 3/15/2016
 * Function to apply to Strings to create labeled points
 */
public class StringToLabeledPoint implements Function<String, LabeledPoint>, Serializable {
  public LabeledPoint call(String line) {
    String[] parts = line.split(" ");
    double[] v = new double[parts.length - 1];
    for (int i = 1; i < parts.length - 1; i++) {
      v[i - 1] = Double.parseDouble(parts[i].split(":")[1]);
    }
    return new LabeledPoint(Double.parseDouble(parts[0]), Vectors.dense(v));
  }
}
