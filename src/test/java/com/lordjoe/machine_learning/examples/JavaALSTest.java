package com.lordjoe.machine_learning.examples;

/**
 * com.lordjoe.machine_learning.JavaALS
 * User: Steve
 * Date: 3/8/2016
 */

import com.lordjoe.distributed.SparkUtilities;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;

import java.util.Arrays;
import java.util.regex.Pattern;

import org.junit.Test;
import scala.Option;
import scala.Tuple2;

/**
 * Example using MLlib ALS from Java.
 */
public final class JavaALSTest {

    static class ParseRating implements Function<String, Rating> {
        private static final Pattern COMMA = Pattern.compile(",");

        @Override
        public Rating call(String line) {
            try {
                String[] tok = COMMA.split(line);
                int x = Integer.parseInt(tok[0]);
                int y = Integer.parseInt(tok[1]);
                double rating = Double.parseDouble(tok[2]);
                return new Rating(x, y, rating);
            } catch (NumberFormatException e) {
                throw new RuntimeException(e);

            }
        }
    }

    static class FeaturesToString implements Function<Tuple2<Object, double[]>, String> {
        @Override
        public String call(Tuple2<Object, double[]> element) {
            return element._1() + "," + Arrays.toString(element._2());
        }
    }

    public static final int DEFAULT_RANK = 4;
    public static final int DEFAULT_ITERATIONS = 4;
    public static final String DEFAULT_FILE = "data/mllib/als/test.data";
    public static final String DEFAULT_OUTPUT = "target/tmp/ALS";

   @Test
    public void testALS()
    {

        int rank = DEFAULT_RANK;
        int iterations = DEFAULT_ITERATIONS;
        String outputDir = DEFAULT_OUTPUT;
        int blocks = -1;

        runALS(DEFAULT_RANK,DEFAULT_RANK,DEFAULT_OUTPUT,blocks,DEFAULT_FILE);
    }

    public static void runALS(int rank ,int iterations,String outputDir,int blocks,String textFile)
    {
        SparkUtilities.setAppName("JavaALS");
        JavaSparkContext  sc = SparkUtilities.getCurrentContext();

        JavaRDD<String> lines = sc.textFile(textFile);

        JavaRDD<Rating> ratings = lines.map(new ParseRating());

        MatrixFactorizationModel model = ALS.train(ratings.rdd(), rank, iterations, 0.01, blocks);

        model.userFeatures().toJavaRDD().map(new FeaturesToString()).saveAsTextFile(
                outputDir + "/userFeatures");
        model.productFeatures().toJavaRDD().map(new FeaturesToString()).saveAsTextFile(
                outputDir + "/productFeatures");
        System.out.println("Final user/product features written to " + outputDir);

    }
//    public static void main(String[] args) {
//
//        if (args.length < 4) {
//            System.err.println(
//                    "Usage: JavaALS <ratings_file> <rank> <iterations> <output_dir> [<blocks>]");
//            System.exit(1);
//        }
//        SparkConf sparkConf = new SparkConf().setAppName("JavaALS");
//        Option<String> option = sparkConf.getOption("spark.master");
//        if (!option.isDefined())    // use local over nothing
//            sparkConf.setMaster("local[*]");
//
//        int rank = Integer.parseInt(args[1]);
//        int iterations = Integer.parseInt(args[2]);
//        String outputDir = args[3];
//        int blocks = -1;
//        if (args.length == 5) {
//            blocks = Integer.parseInt(args[4]);
//        }
//
//        JavaSparkContext sc = new JavaSparkContext(sparkConf);
//
//        runKMeans(rank,iterations,outputDir,blocks,args[0]);
//
//        sc.stop();
//    }
}
