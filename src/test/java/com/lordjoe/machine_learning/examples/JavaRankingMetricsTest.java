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
import java.util.*;

import com.lordjoe.distributed.SparkUtilities;
import org.junit.Test;
import scala.Option;
import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.mllib.evaluation.RankingMetrics;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
// $example off$
import org.apache.spark.SparkConf;

public class JavaRankingMetricsTest {

    @Test
    public void testRanklingMetrics() {
        SparkUtilities.setAppName("Java Ranking Metrics Example");
        JavaSparkContext  sc = SparkUtilities.getCurrentContext();

       // $example on$
    String path = "data/mllib/sample_movielens_data.txt";
    JavaRDD<String> data = sc.textFile(path);
    JavaRDD<Rating> ratings = data.map(
            new StringRatingFunction()
    );
    ratings.cache();

    // Train an ALS model
    final MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), 10, 10, 0.01);

    // Get top 10 recommendations for every user and scale ratings from 0 to 1
    JavaRDD<Tuple2<Object, Rating[]>> userRecs = model.recommendProductsForUsers(10).toJavaRDD();
    JavaRDD<Tuple2<Object, Rating[]>> userRecsScaled = userRecs.map(
            new Tuple2Tuple2Function()
    );
    JavaPairRDD<Object, Rating[]> userRecommended = JavaPairRDD.fromJavaRDD(userRecsScaled);

    // Map ratings to 1 or 0, 1 indicating a movie that should be recommended
    JavaRDD<Rating> binarizedRatings = ratings.map(
            new RatingRatingFunction()
    );

    // Group ratings by common user
    JavaPairRDD<Object, Iterable<Rating>> userMovies = binarizedRatings.groupBy(
            new RatingFunction()
    );

    // Get true relevant documents from all user ratings
    JavaPairRDD<Object, List<Integer>> userMoviesList = userMovies.mapValues(
            new IterableListFunction()
    );

    // Extract the product id from each recommendation
    JavaPairRDD<Object, List<Integer>> userRecommendedList = userRecommended.mapValues(
            new RatingListFunction()
    );
    JavaRDD<Tuple2<List<Integer>, List<Integer>>> relevantDocs = userMoviesList.join(
      userRecommendedList).values();

    // Instantiate the metrics object
    RankingMetrics<Integer> metrics = RankingMetrics.of(relevantDocs);

    // Precision and NDCG at k
    Integer[] kVector = {1, 3, 5};
    for (Integer k : kVector) {
      System.out.format("Precision at %d = %f\n", k, metrics.precisionAt(k));
      System.out.format("NDCG at %d = %f\n", k, metrics.ndcgAt(k));
    }

    // Mean average precision
    System.out.format("Mean average precision = %f\n", metrics.meanAveragePrecision());

    // Evaluate the model using numerical ratings and regression metrics
    JavaRDD<Tuple2<Object, Object>> userProducts = ratings.map(
            new RatingTuple2Function2()
    );
    JavaPairRDD<Tuple2<Integer, Integer>, Object> predictions = JavaPairRDD.fromJavaRDD(
      model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD().map(
              new RatingTuple2FunctionXX()
      ));
    JavaRDD<Tuple2<Object, Object>> ratesAndPreds =
      JavaPairRDD.fromJavaRDD(ratings.map(
              new RatingTuple2Function()
      )).join(predictions).values();

    // Create regression metrics object
    RegressionMetrics regressionMetrics = new RegressionMetrics(ratesAndPreds.rdd());

    // Root mean squared error
    System.out.format("RMSE = %f\n", regressionMetrics.rootMeanSquaredError());

    // R-squared
    System.out.format("R-squared = %f\n", regressionMetrics.r2());
    // $example off$
  }

    private static class StringRatingFunction implements Function<String, Rating> ,Serializable {
        @Override
        public Rating call(String line) {
          String[] parts = line.split("::");
            return new Rating(Integer.parseInt(parts[0]), Integer.parseInt(parts[1]), Double
              .parseDouble(parts[2]) - 2.5);
        }
    }

    private static class RatingTuple2Function implements Function<Rating, Tuple2<Tuple2<Integer, Integer>, Object>> ,Serializable {
        @Override
        public Tuple2<Tuple2<Integer, Integer>, Object> call(Rating r) {
          return new Tuple2<Tuple2<Integer, Integer>, Object>(
            new Tuple2<>(r.user(), r.product()), r.rating());
        }
    }

    private static class RatingTuple2FunctionXX implements Function<Rating, Tuple2<Tuple2<Integer, Integer>, Object>>,Serializable  {
        @Override
        public Tuple2<Tuple2<Integer, Integer>, Object> call(Rating r) {
          return new Tuple2<Tuple2<Integer, Integer>, Object>(
            new Tuple2<>(r.user(), r.product()), r.rating());
        }
    }

    private static class Tuple2Tuple2Function implements Function<Tuple2<Object, Rating[]>, Tuple2<Object, Rating[]>>,Serializable  {
        @Override
        public Tuple2<Object, Rating[]> call(Tuple2<Object, Rating[]> t) {
          Rating[] scaledRatings = new Rating[t._2().length];
          for (int i = 0; i < scaledRatings.length; i++) {
            double newRating = Math.max(Math.min(t._2()[i].rating(), 1.0), 0.0);
            scaledRatings[i] = new Rating(t._2()[i].user(), t._2()[i].product(), newRating);
          }
          return new Tuple2<>(t._1(), scaledRatings);
        }
    }

    private static class RatingRatingFunction implements Function<Rating, Rating>,Serializable  {
        @Override
        public Rating call(Rating r) {
          double binaryRating;
          if (r.rating() > 0.0) {
            binaryRating = 1.0;
          } else {
            binaryRating = 0.0;
          }
          return new Rating(r.user(), r.product(), binaryRating);
        }
    }

    private static class RatingFunction implements Function<Rating, Object> ,Serializable {
        @Override
        public Object call(Rating r) {
          return r.user();
        }
    }

    private static class IterableListFunction implements Function<Iterable<Rating>, List<Integer>>,Serializable  {
        @Override
        public List<Integer> call(Iterable<Rating> docs) {
          List<Integer> products = new ArrayList<>();
          for (Rating r : docs) {
            if (r.rating() > 0.0) {
              products.add(r.product());
            }
          }
          return products;
        }
    }

    private static class RatingListFunction implements Function<Rating[], List<Integer>>,Serializable  {
        @Override
        public List<Integer> call(Rating[] docs) {
          List<Integer> products = new ArrayList<>();
          for (Rating r : docs) {
            products.add(r.product());
          }
          return products;
        }
    }

    private static class RatingTuple2Function2 implements Function<Rating, Tuple2<Object, Object>>,Serializable  {
        @Override
        public Tuple2<Object, Object> call(Rating r) {
          return new Tuple2<Object, Object>(r.user(), r.product());
        }
    }
}
