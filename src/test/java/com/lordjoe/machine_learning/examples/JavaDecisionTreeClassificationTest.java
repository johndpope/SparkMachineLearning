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
import org.apache.spark.SparkContext;
import org.junit.Test;
import scala.Option;
import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
// $example off$

public class JavaDecisionTreeClassificationTest {

    @Test
    public void testDecisionTreeClassification() {
        SparkUtilities.setAppName("JavaDecisionTreeClassificationExample");
        JavaSparkContext jsc = SparkUtilities.getCurrentContext();

        // Load and parse the data file.
        String datapath = "data/mllib/sample_libsvm_data.txt";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
        // Split the data into training and test sets (30% held out for testing)
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        // Set parameters.
        //  Empty categoricalFeaturesInfo indicates all features are continuous.
        Integer numClasses = 2;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        String impurity = "gini";
        Integer maxDepth = 5;
        Integer maxBins = 32;

        // Train a DecisionTree model for classification.
        final DecisionTreeModel model = DecisionTree.trainClassifier(trainingData, numClasses,
                categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        // Evaluate model on test instances and compute test error
        JavaPairRDD<Double, Double> predictionAndLabel =
                testData.mapToPair(new LabeledPointDoubleDoublePairFunction(model));
        Double testErr =
                1.0 * predictionAndLabel.filter(new Tuple2BooleanFunction()).count() / testData.count();

        System.out.println("Test Error: " + testErr);
        System.out.println("Learned classification tree model:\n" + model.toDebugString());

        // Save and load model

        SparkUtilities.saveModel(model, "target/tmp/myDecisionTreeClassificationModel");

        DecisionTreeModel sameModel = DecisionTreeModel
                .load(jsc.sc(), "target/tmp/myDecisionTreeClassificationModel");
        // $example off$
    }

    private static class LabeledPointDoubleDoublePairFunction implements PairFunction<LabeledPoint, Double, Double>,Serializable {
        private final DecisionTreeModel m_model;

        public LabeledPointDoubleDoublePairFunction(DecisionTreeModel model) {
            m_model = model;
        }

        @Override
        public Tuple2<Double, Double> call(LabeledPoint p) {
            return new Tuple2<Double, Double>(m_model.predict(p.features()), p.label());
        }
    }

    private static class Tuple2BooleanFunction implements Function<Tuple2<Double, Double>, Boolean>,Serializable {
        @Override
        public Boolean call(Tuple2<Double, Double> pl) {
            return !pl._1().equals(pl._2());
        }
    }
}
