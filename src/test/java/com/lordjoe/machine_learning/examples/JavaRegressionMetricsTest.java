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

// $example on$
import com.lordjoe.distributed.SparkUtilities;
import com.lordjoe.machine_learning.StringToLabeledPoint;
import org.junit.Test;
import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.mllib.evaluation.RegressionMetrics;

import java.io.Serializable;
// $example off$

public class JavaRegressionMetricsTest {

  @Test
  public void testRegressionMetrics() {
    SparkUtilities.setAppName("Java Regression Metrics Example");
    JavaSparkContext sc = SparkUtilities.getCurrentContext();

       // $example on$
    // Load and parse the data
    String path = "data/mllib/sample_linear_regression_data.txt";
    JavaRDD<String> data = sc.textFile(path);
    JavaRDD<LabeledPoint> parsedData = data.map(
            new StringToLabeledPoint()
    );
    parsedData.cache();

    // Building the model
    int numIterations = 100;
    final LinearRegressionModel model = LinearRegressionWithSGD.train(JavaRDD.toRDD(parsedData),
      numIterations);

    // Evaluate model on training examples and compute training error
    JavaRDD<Tuple2<Object, Object>> valuesAndPreds = parsedData.map(
            new LabeledPointTuple2Function(model)
    );

    // Instantiate metrics object
    RegressionMetrics metrics = new RegressionMetrics(valuesAndPreds.rdd());

    // Squared error
    System.out.format("MSE = %f\n", metrics.meanSquaredError());
    System.out.format("RMSE = %f\n", metrics.rootMeanSquaredError());

    // R-squared
    System.out.format("R Squared = %f\n", metrics.r2());

    // Mean absolute error
    System.out.format("MAE = %f\n", metrics.meanAbsoluteError());

    // Explained variance
    System.out.format("Explained Variance = %f\n", metrics.explainedVariance());

    // Save and load model
    SparkUtilities.saveModel(model,  "target/tmp/LogisticRegressionModel");
    LinearRegressionModel sameModel = LinearRegressionModel.load(sc.sc(),
      "target/tmp/LogisticRegressionModel");
    // $example off$
  }

  private static class LabeledPointTuple2Function implements Function<LabeledPoint, Tuple2<Object, Object>>,Serializable  {
    private final LinearRegressionModel m_model;

    public LabeledPointTuple2Function(LinearRegressionModel model) {
      m_model = model;
    }

    public Tuple2<Object, Object> call(LabeledPoint point) {
      double prediction = m_model.predict(point.features());
      return new Tuple2<Object, Object>(prediction, point.label());
    }
  }
}
