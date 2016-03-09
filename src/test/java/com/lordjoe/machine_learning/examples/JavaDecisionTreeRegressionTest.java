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
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import com.lordjoe.distributed.SparkUtilities;
import org.junit.Test;
import scala.Tuple2;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
// $example off$

public class JavaDecisionTreeRegressionTest {

  @Test
  public  void testDecisionTreeRegression() {
    SparkUtilities.setAppName("JavaDecisionTreeRegressionExample");
    JavaSparkContext jsc = SparkUtilities.getCurrentContext();

    // Load and parse the data file.
    String datapath = "data/mllib/sample_libsvm_data.txt";
    JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
    // Split the data into training and test sets (30% held out for testing)
    JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
    JavaRDD<LabeledPoint> trainingData = splits[0];
    JavaRDD<LabeledPoint> testData = splits[1];

    // Set parameters.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
    String impurity = "variance";
    Integer maxDepth = 5;
    Integer maxBins = 32;

    // Train a DecisionTree model.
    final DecisionTreeModel model = DecisionTree.trainRegressor(trainingData,
      categoricalFeaturesInfo, impurity, maxDepth, maxBins);

    // Evaluate model on test instances and compute test error
    JavaPairRDD<Double, Double> predictionAndLabel =
      testData.mapToPair(new DoPredict(model));
    Double testMSE =
      predictionAndLabel.map(new SquaredDifference()).reduce(new DoubleDoubleDoubleFunction2()) / data.count();
    System.out.println("Test Mean Squared Error: " + testMSE);
    System.out.println("Learned regression tree model:\n" + model.toDebugString());

    // Save and load model
      SparkUtilities.saveModel(model, "target/tmp/myDecisionTreeRegressionModel");
    DecisionTreeModel sameModel = DecisionTreeModel
      .load(jsc.sc(), "target/tmp/myDecisionTreeRegressionModel");
    // $example off$
  }

    private static class DoPredict implements PairFunction<LabeledPoint, Double, Double>,Serializable{
        private final DecisionTreeModel m_model;

        public DoPredict(DecisionTreeModel model) {
            m_model = model;
        }

        @Override
        public Tuple2<Double, Double> call(LabeledPoint p) {
          return new Tuple2<Double, Double>(m_model.predict(p.features()), p.label());
        }
    }

    private static class SquaredDifference implements Function<Tuple2<Double, Double>, Double>,Serializable {
        @Override
        public Double call(Tuple2<Double, Double> pl) {
          Double diff = pl._1() - pl._2();
          return diff * diff;
        }
    }

    private static class DoubleDoubleDoubleFunction2 implements Function2<Double, Double, Double>,Serializable {
        @Override
        public Double call(Double a, Double b) {
          return a + b;
        }
    }
}
