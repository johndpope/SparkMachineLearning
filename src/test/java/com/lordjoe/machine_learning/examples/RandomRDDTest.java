package com.lordjoe.machine_learning.examples;

import akka.pattern.AskTimeoutException;
import com.lordjoe.distributed.SparkUtilities;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.fpm.AssociationRules;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.random.RandomRDDs;
import org.apache.spark.mllib.rdd.RandomRDD;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.stat.Statistics;
import org.junit.*;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * com.lordjoe.machine_learning.examples.RandomRDDTest
 * User: Steve
 * Date: 3/9/2016
 */
public class RandomRDDTest {

    public static final int SAMPLE_SIZE = 10000000;

    @Test
    public void testBuildCorrelation() throws Exception {
        SparkUtilities.setAppName("JavaAssociationRulesExample");
        JavaSparkContext sc = SparkUtilities.getCurrentContext();
        double mean = 10;
        double sd = 3;

        double correlation = 0.7;

        JavaDoubleRDD normalRdd = RandomRDDs.normalJavaRDD(sc, SAMPLE_SIZE);
        JavaRDD<Double> map = normalRdd.map(new MeanSd(10.0, 3.0));
        JavaRDD<Vector> correleted = correleted(normalRdd, correlation);
         assertCorrelated(correleted,correlation,SAMPLE_SIZE);
    }

    public static void assertUncorrelated(JavaRDD<Vector> test, long sampleSize) {
        double err = 10 * (1.0 / Math.sqrt(sampleSize));
        Matrix corr = Statistics.corr(test.rdd());
        for (int i = 0; i < corr.numCols(); i++) {
            for (int j = 0; j < corr.numRows(); j++) {
                if (i == j) {
                    double errx = Math.abs(1.0 - corr.apply(i, j));
                    Assert.assertTrue(errx < err);
                } else {
                    double errx = Math.abs(0.0 - corr.apply(i, j));
                    Assert.assertTrue(errx < err);
                }

            }
        }
    }

    public static void assertCorrelated(JavaRDD<Vector> test,double correletion, long sampleSize) {
        double err = 1.0 / Math.sqrt(sampleSize);
        Matrix corr = Statistics.corr(test.rdd());
        for (int i = 0; i < corr.numCols(); i++) {
            for (int j = 0; j < corr.numRows(); j++) {
                if (i == j) {
                    double errx = Math.abs(1.0 - corr.apply(i, j));
                    Assert.assertTrue(errx < err);
                } else {
                    double errx = Math.abs(0.0 - corr.apply(i, j));
                    Assert.assertTrue(errx < correletion);
                }
            }
        }
    }


    @Test
    public void testCorrelationOf1() throws Exception {
        SparkUtilities.setAppName("JavaAssociationRulesExample");
        JavaSparkContext sc = SparkUtilities.getCurrentContext();
        double mean = 10;
        double sd = 3;

        JavaRDD<Vector> vectorJavaRDD = RandomRDDs.normalJavaVectorRDD(sc, SAMPLE_SIZE, 3);
        assertUncorrelated(vectorJavaRDD,SAMPLE_SIZE);

        Matrix corr = Statistics.corr(vectorJavaRDD.rdd());
        printMatrix(corr);
    }


    public static void printMatrix(Matrix corr) {
        for (int i = 0; i < corr.numCols(); i++) {
            for (int j = 0; j < corr.numRows(); j++) {
                System.out.print(String.format("%8.3f", corr.apply(i, j)));
                System.out.println(" ");
            }
            System.out.println("\n");
        }
    }

    public static JavaRDD<Vector> correleted(JavaDoubleRDD normalRdd, double correlation) {
        JavaSparkContext sc = SparkUtilities.getCurrentContext();
        JavaRDD<Vector> ret = normalRdd.map(new Correlation(10.0, 3.0, correlation));

        return ret;
    }


    public static JavaRDD<Vector> correleted(JavaRDD<Double> normalRdd, double correlation) {
        JavaSparkContext sc = SparkUtilities.getCurrentContext();
        JavaRDD<Vector> ret = normalRdd.map(new Correlation(10.0, 3.0, correlation));
        return ret;
    }

    public static final Random RND = new Random();

    /**
     * convert normalRDD to normal with mean mean and Sd sd
     */
    public static class Correlation implements Function<Double, Vector>, Serializable {
        private final double mean;
        private final double sd;
        private final double corr;
        private final double xcorr;

        public Correlation(double mean, double sd, double corr) {
            this.mean = mean;
            this.sd = sd;
            this.corr = corr;
            xcorr = Math.sqrt(1.0 * corr * corr);
        }

        @Override
        public Vector call(Double v1) throws Exception {
            double x = (v1 * sd) + mean;
            double y = x  + ((xcorr *  RND.nextGaussian()  * sd) + mean);
            Vector dense = Vectors.dense(x, y);
            return dense;
        }
    }

    /**
     * convert normalRDD to normal with mean mean and Sd sd
     */
    public static class MeanSd implements Function<Double, Double>, Serializable {
        private final double mean;
        private final double sd;

        public MeanSd(double mean, double sd) {
            this.mean = mean;
            this.sd = sd;
        }

        @Override
        public Double call(Double v1) throws Exception {
            return (v1 * sd) + mean;
        }
    }
}
