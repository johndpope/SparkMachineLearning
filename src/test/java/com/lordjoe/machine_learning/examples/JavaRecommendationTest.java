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
import scala.Option;
import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.SparkConf;

import java.io.Serializable;
// $example off$

public class JavaRecommendationTest {

    @Test
    public void testRecommendation() {
        SparkUtilities.setAppName("Java Collaborative Filtering Example");
        JavaSparkContext jsc = SparkUtilities.getCurrentContext();


        // Load and parse the data
        String path = "data/mllib/als/test.data";
        JavaRDD<String> data = jsc.textFile(path);
        JavaRDD<Rating> ratings = data.map(
                new StringRatingFunction()
        );

        // Build the recommendation model using ALS
        int rank = 10;
        int numIterations = 10;
        MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations, 0.01);

        // Evaluate the model on rating data
        JavaRDD<Tuple2<Object, Object>> userProducts = ratings.map(
                new RatingTuple2Function()
        );
        JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
                model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD().map(
                        new RatingTuple2Function2()
                ));
        JavaRDD<Tuple2<Double, Double>> ratesAndPreds =
                JavaPairRDD.fromJavaRDD(ratings.map(
                        new RatingTuple2Function3()
                )).join(predictions).values();
        double MSE = JavaDoubleRDD.fromRDD(ratesAndPreds.map(
                new Tuple2Function()
        ).rdd()).mean();
        System.out.println("Mean Squared Error = " + MSE);

        // Save and load model
        SparkUtilities.saveModel(model,  "target/tmp/myCollaborativeFilter");
        MatrixFactorizationModel sameModel = MatrixFactorizationModel.load(jsc.sc(),
                "target/tmp/myCollaborativeFilter");
        // $example off$
    }

    private static class StringRatingFunction implements Function<String, Rating>,Serializable  {
        public Rating call(String s) {
            String[] sarray = s.split(",");
            return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]),
                    Double.parseDouble(sarray[2]));
        }
    }

    private static class RatingTuple2Function implements Function<Rating, Tuple2<Object, Object>>,Serializable  {
        public Tuple2<Object, Object> call(Rating r) {
            return new Tuple2<Object, Object>(r.user(), r.product());
        }
    }

    private static class RatingTuple2Function2 implements Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>,Serializable  {
        public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r) {
            return new Tuple2<Tuple2<Integer, Integer>, Double>(
                    new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
        }
    }

    private static class RatingTuple2Function3 implements Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>,Serializable  {
        public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r) {
            return new Tuple2<Tuple2<Integer, Integer>, Double>(
                    new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
        }
    }

    private static class Tuple2Function implements Function<Tuple2<Double, Double>, Object>,Serializable  {
        public Object call(Tuple2<Double, Double> pair) {
            Double err = pair._1() - pair._2();
            return err * err;
        }
    }
}
