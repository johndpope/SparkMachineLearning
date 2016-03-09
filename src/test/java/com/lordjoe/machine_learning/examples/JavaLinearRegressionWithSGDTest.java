/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.lordjoe.machine_learning.examples;

import com.lordjoe.distributed.SparkUtilities;
import org.apache.spark.api.java.JavaSparkContext;

// $example on$
import org.junit.Test;
import scala.Tuple2;

import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;

import java.io.Serializable;
// $example off$

/**
 * Example for LinearRegressionWithSGD.
 */
public class JavaLinearRegressionWithSGDTest {
  @Test
  public void testLinearRegressionWithSGD() {
    SparkUtilities.setAppName("JavaLinearRegressionWithSGDExample");
    JavaSparkContext sc = SparkUtilities.getCurrentContext();

    // $example on$
    // Load and parse the data
    String path = "data/mllib/ridge-data/lpsa.data";
    JavaRDD<String> data = sc.textFile(path);
    JavaRDD<LabeledPoint> parsedData = data.map(
            new SplitString()
    );
    parsedData.cache();

    // Building the model
    int numIterations = 100;
    double stepSize = 0.00000001;
    final LinearRegressionModel model =
      LinearRegressionWithSGD.train(JavaRDD.toRDD(parsedData), numIterations, stepSize);

    // Evaluate model on training examples and compute training error
    JavaRDD<Tuple2<Double, Double>> valuesAndPreds = parsedData.map(
            new DoPredict(model)
    );
    double MSE = new JavaDoubleRDD(valuesAndPreds.map(
            new Distance()
    ).rdd()).mean();
    System.out.println("training Mean Squared Error = " + MSE);

    // Save and load model
    SparkUtilities.saveModel(model,  "target/tmp/javaLinearRegressionWithSGDModel");
    LinearRegressionModel sameModel = LinearRegressionModel.load(sc.sc(),
      "target/tmp/javaLinearRegressionWithSGDModel");
    // $example off$

    //sc.stop();
  }

  private static class SplitString implements Function<String, LabeledPoint>,Serializable {
    public LabeledPoint call(String line) {
      String[] parts = line.split(",");
      String[] features = parts[1].split(" ");
      double[] v = new double[features.length];
      for (int i = 0; i < features.length - 1; i++) {
        v[i] = Double.parseDouble(features[i]);
      }
      return new LabeledPoint(Double.parseDouble(parts[0]), Vectors.dense(v));
    }
  }

  private static class DoPredict implements Function<LabeledPoint, Tuple2<Double, Double>>,Serializable  {
    private final LinearRegressionModel m_model;

    public DoPredict(LinearRegressionModel model) {
      m_model = model;
    }

    public Tuple2<Double, Double> call(LabeledPoint point) {
      double prediction = m_model.predict(point.features());
      return new Tuple2<Double, Double>(prediction, point.label());
    }
  }

  private static class Distance implements Function<Tuple2<Double, Double>, Object>,Serializable  {
    public Object call(Tuple2<Double, Double> pair) {
      return Math.pow(pair._1() - pair._2(), 2.0);
    }
  }
}
