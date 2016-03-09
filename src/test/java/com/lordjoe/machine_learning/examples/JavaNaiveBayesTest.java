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
import org.junit.Test;
import scala.Tuple2;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;
// $example off$

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class JavaNaiveBayesTest {
    @Test
    public void testLNaiveBayes() {
        SparkUtilities.setAppName("JavaNaiveBayesExample");
        JavaSparkContext jsc = SparkUtilities.getCurrentContext();

        // $example on$
        String path = "data/mllib/sample_naive_bayes_data.txt";

        List<LabeledPoint> labeledPoints = readLabeledPoints(path);
        JavaRDD<LabeledPoint> inputData = jsc.parallelize(labeledPoints);

        JavaRDD<LabeledPoint>[] tmp = inputData.randomSplit(new double[]{0.6, 0.4}, 12345);
        JavaRDD<LabeledPoint> training = tmp[0]; // training set
        JavaRDD<LabeledPoint> test = tmp[1]; // test set
        final NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
        JavaPairRDD<Double, Double> predictionAndLabel =
                test.mapToPair(new DoPredict(model));
        double accuracy = predictionAndLabel.filter(new Tuple2BooleanFunction()).count() / (double) test.count();

        // Save and load model
        SparkUtilities.saveModel(model, "target/tmp/myNaiveBayesModel");
        NaiveBayesModel sameModel = NaiveBayesModel.load(jsc.sc(), "target/tmp/myNaiveBayesModel");
        // $example off$
    }

    public static List<LabeledPoint> readLabeledPoints(String path) {
        List<LabeledPoint> holder = new ArrayList<LabeledPoint>();
        try {
            FileReader rdr = new FileReader(path);
            LineNumberReader inp = new LineNumberReader(rdr);
            String line = inp.readLine();
            while (line != null) {
                LabeledPoint e = LabeledPoint.parse(line);
                holder.add(e);
                line = inp.readLine();
            }
            return holder;
        } catch (IOException e) {
            throw new RuntimeException(e);

        }
    }

    private static class Tuple2BooleanFunction implements Function<Tuple2<Double, Double>, Boolean>, Serializable {
        @Override
        public Boolean call(Tuple2<Double, Double> pl) {
            return pl._1().equals(pl._2());
        }
    }

    private static class DoPredict implements PairFunction<LabeledPoint, Double, Double>,Serializable {
        private final NaiveBayesModel m_model;

        public DoPredict(NaiveBayesModel model) {
            m_model = model;
        }

        @Override
        public Tuple2<Double, Double> call(LabeledPoint p) {
            return new Tuple2<Double, Double>(m_model.predict(p.features()), p.label());
        }
    }
}
