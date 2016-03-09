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
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;

// $example on$
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.Test;
import scala.Option;
import scala.Tuple2;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import java.io.Serializable;
// $example off$

/**
 * Example for SVMWithSGD.
 */
public class JavaSVMWithSGDTest {

  @Test
  public void testSVD() {

    SparkUtilities.setAppName("JavaSVMWithSGDExample");
      JavaSparkContext sc = SparkUtilities.getCurrentContext();

       // $example on$
    String path = "data/mllib/sample_libsvm_data.txt";
    JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), path).toJavaRDD();

    // Split initial RDD into two... [60% training data, 40% testing data].
    JavaRDD<LabeledPoint> training = data.sample(false, 0.6, 11L);
    training.cache();
    JavaRDD<LabeledPoint> test = data.subtract(training);

    // Run training algorithm to build the model.
    int numIterations = 100;
    final SVMModel model = SVMWithSGD.train(training.rdd(), numIterations);

    // Clear the default threshold.
    model.clearThreshold();

    // Compute raw scores on the test set.
    JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test.map(
            new LabeledPointTuple2Function(model)
    );

    // Get evaluation metrics.
    BinaryClassificationMetrics metrics =
      new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
    double auROC = metrics.areaUnderROC();

    System.out.println("Area under ROC = " + auROC);

    // Save and load model
      SparkUtilities.saveModel(model,  "target/tmp/javaSVMWithSGDModel");
    SVMModel sameModel = SVMModel.load(sc.sc(), "target/tmp/javaSVMWithSGDModel");
    // $example off$

    //sc.stop();
  }

    private static class LabeledPointTuple2Function implements Function<LabeledPoint, Tuple2<Object, Object>>,Serializable {
        private final SVMModel m_model;

        public LabeledPointTuple2Function(SVMModel model) {
            m_model = model;
        }

        public Tuple2<Object, Object> call(LabeledPoint p) {
          Double score = m_model.predict(p.features());
          return new Tuple2<Object, Object>(score, p.label());
        }
    }
}
