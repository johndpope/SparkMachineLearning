package com.lordjoe.machine_learning.binary_classification;

import com.lordjoe.distributed.SparkUtilities;
import com.lordjoe.machine_learning.LabeledPointUtilities;
import com.lordjoe.testing.ElapsedTimer;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * com.lordjoe.machine_learning.JavaDCExampleTest
 * User: Steve
 * Date: 3/21/2016
 */
public class ImageGradientBoostedTree {

    public static final int NUMBER_CLASSES = 2;
    public static final double TRAINING_SET_FRACTION = 0.7;
    public static final double TEST_SET_FRACTION = 1.0 - TRAINING_SET_FRACTION;
    public static final long DEFAULT_SEED = 11L;
    public static final int MINST_IMAGE_WIDTH = 28;
    public static final int MINST_IMAGE_HEIGHT = 28;
    public static final String DEFAULT_FILE = "MNIST_Digits/train58LabeledPoints.txt";


    public static void run(String inputFile,  int iterations, int depth) {
        SparkUtilities.setAppName("logisticRegressionWithLBFGS");
        JavaSparkContext sc = SparkUtilities.getCurrentContext();
        JavaRDD<String> lines = sc.textFile(inputFile);

        JavaRDD<LabeledPoint> labeledPoints = lines.map(new LabeledPointUtilities.ParseLabeledPoint());

        // remap labels 5 and 8 to 0 and 1
        double[] labelsToRemap = { 5.0,8.0 };
        labeledPoints = BinaryClassificationUtilities.remapLabels(labeledPoints,labelsToRemap);


        //      labeledPoints = labeledPoints.map(new GaborWaveletFeature(MINST_IMAGE_WIDTH,MINST_IMAGE_HEIGHT) );
        //     labeledPoints = labeledPoints.map(new HoughTransform(MINST_IMAGE_WIDTH,MINST_IMAGE_HEIGHT) );

        // Split initial RDD into two... [60% training data, 40% testing data].
        JavaRDD<LabeledPoint>[] splits = labeledPoints.randomSplit(new double[]{TRAINING_SET_FRACTION, TEST_SET_FRACTION}, DEFAULT_SEED);
        JavaRDD<LabeledPoint> trainingData = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1];


        // Train a GradientBoostedTrees model.
// The defaultParams for Classification use LogLoss by default.
        BoostingStrategy boostingStrategy = BoostingStrategy.defaultParams("Classification");
        boostingStrategy.setNumIterations(iterations);
        boostingStrategy.getTreeStrategy().setNumClasses(NUMBER_CLASSES);
        boostingStrategy.getTreeStrategy().setMaxDepth(depth);
// Empty categoricalFeaturesInfo indicates all features are continuous.
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        boostingStrategy.treeStrategy().setCategoricalFeaturesInfo(categoricalFeaturesInfo);

        ElapsedTimer timer = new ElapsedTimer();
        final GradientBoostedTreesModel model =
                GradientBoostedTrees.train(trainingData, boostingStrategy);

        timer.showElapsed("Time for " + SparkUtilities.getAppName() + " Training");


        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = test.map(
                new DoPredict(model)
        );

        BinaryClassificationUtilities.showAllStatistics(predictionAndLabels);

    }

    private static class DoPredict implements Function<LabeledPoint, Tuple2<Object, Object>>, Serializable {
        private final GradientBoostedTreesModel m_model;

        public DoPredict(GradientBoostedTreesModel model) {
            m_model = model;
        }

        public Tuple2<Object, Object> call(LabeledPoint p) {
            Double prediction = m_model.predict(p.features());
            return new Tuple2<Object, Object>(prediction, p.label());
        }
    }

      /**
     * Use base directory of   "Spark Machine Learning
     *
     * @param args
     */
    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println(
                    "Usage: public class ImageGradientBoostedTree <input_file>   <max_iterations> [<depth>]");
            System.exit(1);
        }

        Logger.getLogger("org").setLevel(Level.WARN);
        Logger.getLogger("akka").setLevel(Level.WARN);

        String inputFile = args[0];
            int iterations = Integer.parseInt(args[1]);
        int depth = 8;

        if (args.length >= 4) {
            depth = Integer.parseInt(args[2]);
        }

        run(inputFile,   iterations, depth);

        //sc.stop();
    }
}