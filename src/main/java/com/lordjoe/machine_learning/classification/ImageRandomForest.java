package com.lordjoe.machine_learning.classification;

import com.lordjoe.distributed.SparkUtilities;
import com.lordjoe.machine_learning.LabeledPointUtilities;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import scala.Tuple2;

import java.io.Serializable;
import java.util.HashMap;

/**
 * com.lordjoe.machine_learning.ImageSupervised
 * User: Steve
 * Date: 3/21/2016
 */
public class ImageRandomForest {


    public static final int MINST_IMAGE_WIDTH = 28;
    public static final int MINST_IMAGE_HEIGHT = 28 ;
    public static final int MINST_IMAGE_SIZE = MINST_IMAGE_WIDTH *  MINST_IMAGE_HEIGHT;

    public static final int DEFAULT_K = 4;
    public static final int DEFAULT_ITERATIONS = 4;
    public static final int DEFAULT_RUNS = 1;
   // public static final String DEFAULT_FILE = "MNIST_Digits/t10kLabeledPoints.txt";
    public static final String DEFAULT_FILE = "MNIST_Digits/trainLabeledPoints.txt";


    public static void run(String inputFile, int k, int iterations, int runs) {
        SparkUtilities.setAppName("logisticRegressionWithLBFGS");
        JavaSparkContext sc = SparkUtilities.getCurrentContext();
        JavaRDD<String> lines = sc.textFile(inputFile);

        JavaRDD<LabeledPoint> labeledPoints = lines.map(new LabeledPointUtilities.ParseLabeledPoint());

  //      labeledPoints = labeledPoints.map(new GaborWaveletFeature(MINST_IMAGE_WIDTH,MINST_IMAGE_HEIGHT) );
   //     labeledPoints = labeledPoints.map(new HoughTransform(MINST_IMAGE_WIDTH,MINST_IMAGE_HEIGHT) );

        // Split initial RDD into two... [60% training data, 40% testing data].
        JavaRDD<LabeledPoint>[] splits = labeledPoints.randomSplit(new double[]{0.6, 0.4}, 11L);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1];


        // Run training algorithm to build the model.
        // Train a RandomForest model.
        // Empty categoricalFeaturesInfo indicates all features are continuous.
        Integer numClasses = 10;
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        Integer numTrees = 12; // Use more in practice.
        String featureSubsetStrategy = "auto"; // Let the algorithm choose.
        String impurity = "gini";
        Integer maxDepth = 5;
        Integer maxBins = 32;
        Integer seed = 12345;

        final RandomForestModel model = RandomForest.trainClassifier(training, numClasses,
                categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
                seed);
        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = test.map(
                new DoPredict(model)
        );
        LabeledPointUtilities.showMulticlassMetrics(predictionAndLabels);


//        // Save and load model
//        SparkUtilities.saveModel(model, "target/tmp/LogisticRegressionModel");
//        LogisticRegressionModel sameModel = LogisticRegressionModel.load(sc.sc(),
//                "target/tmp/LogisticRegressionModel");
        // $example off$
    }


    private static class DoPredict implements Function<LabeledPoint, Tuple2<Object, Object>>,Serializable {
        private final RandomForestModel m_model;

        public DoPredict(RandomForestModel model) {
            m_model = model;
        }

        public Tuple2<Object, Object> call(LabeledPoint p) {
            Double prediction = m_model.predict(p.features());
            return new Tuple2<Object, Object>(prediction, p.label());
        }
    }
    private static class Tuple2BooleanFunction implements Function<Tuple2<Double, Double>, Boolean>,Serializable  {
        @Override
        public Boolean call(Tuple2<Double, Double> pl) {
            return !pl._1().equals(pl._2());
        }
    }

    /**
     * Use base directory of   "Spark Machine Learning
     *
     * @param args
     */
    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println(
                    "Usage: public class ImageKMeans <input_file> <k> <max_iterations> [<runs>]");
            System.exit(1);
        }

        Logger.getLogger("org").setLevel(Level.WARN);
        Logger.getLogger("akka").setLevel(Level.WARN);

        String inputFile = args[0];
        int k = Integer.parseInt(args[1]);
        int iterations = Integer.parseInt(args[2]);
        int runs = 1;

        if (args.length >= 4) {
            runs = Integer.parseInt(args[3]);
        }

        run(inputFile, k, iterations, runs);

        //sc.stop();
    }
}