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

// $example on$
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.Test;
import scala.Tuple2;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import java.io.Serializable;
// $example off$

/**
 * Example for LogisticRegressionWithLBFGS.
 */
public class JavaLogisticRegressionWithLBFGSETest {
  @Test
  public void testLogisticRegressionWithSGD() {
    SparkUtilities.setAppName("JavaLogisticRegressionWithLBFGSExample");
    JavaSparkContext jsc = SparkUtilities.getCurrentContext();

      // $example on$
    String path = "data/mllib/sample_libsvm_data.txt";
    JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), path).toJavaRDD();


    // Split initial RDD into two... [60% training data, 40% testing data].
    JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[] {0.6, 0.4}, 11L);
    JavaRDD<LabeledPoint> training = splits[0].cache();
    JavaRDD<LabeledPoint> test = splits[1];

    // Run training algorithm to build the model.
    final LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
      .setNumClasses(10)
      .run(training.rdd());

    // Compute raw scores on the test set.
    JavaRDD<Tuple2<Object, Object>> predictionAndLabels = test.map(
            new DoPredict(model)
    );

    // Get evaluation metrics.
    MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
    double precision = metrics.precision();
    System.out.println("Precision = " + precision);

    // Save and load model
      SparkUtilities.saveModel(model,  "target/tmp/javaLogisticRegressionWithLBFGSModel");
    LogisticRegressionModel sameModel = LogisticRegressionModel.load(jsc.sc(),
      "target/tmp/javaLogisticRegressionWithLBFGSModel");
    // $example off$


  }

    private static class DoPredict implements Function<LabeledPoint, Tuple2<Object, Object>>,Serializable {
        private final LogisticRegressionModel m_model;

        public DoPredict(LogisticRegressionModel model) {
            m_model = model;
        }

        public Tuple2<Object, Object> call(LabeledPoint p) {
          Double prediction = m_model.predict(p.features());
          return new Tuple2<Object, Object>(prediction, p.label());
        }
    }
}
