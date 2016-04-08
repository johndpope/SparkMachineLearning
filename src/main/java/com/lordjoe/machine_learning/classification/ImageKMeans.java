package com.lordjoe.machine_learning.classification;

import com.lordjoe.distributed.SparkUtilities;
import com.lordjoe.machine_learning.LabeledPointUtilities;
import com.lordjoe.machine_learning.image_analysis.HoughLineTransform;
import com.lordjoe.machine_learning.image_analysis.HoughTransform;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.storage.StorageLevel;

import java.io.Serializable;
import java.util.List;
import java.util.regex.Pattern;

/**
 * com.lordjoe.machine_learning.ImageKMeans
 * User: Steve
 * Date: 3/21/2016
 */
public class ImageKMeans {


    public static class Classification implements Serializable {
        public final LabeledPoint pt;
        public final int prediction;
        public final int original_prediction;

        public Classification(LabeledPoint pt, int prediction, int original_prediction) {
            this.pt = pt;
            this.prediction = prediction;
            this.original_prediction = original_prediction;
        }
    }
    public static class ParsePoint3 implements Function<LabeledPoint, Classification>, Serializable  {
        public final KMeansModel model;

        public ParsePoint3(KMeansModel model) {
            this.model = model;
        }

        @Override
        public Classification call(LabeledPoint image) {
            int predict = model.predict(image.features());
            return new Classification(image,predict,(int)image.label());
        }
    }


    public static final int MINST_IMAGE_WIDTH = 28;
    public static final int MINST_IMAGE_HEIGHT = 28 ;
    public static final int MINST_IMAGE_SIZE = MINST_IMAGE_WIDTH *  MINST_IMAGE_HEIGHT;

    public static final int DEFAULT_K = 4;
    public static final int DEFAULT_ITERATIONS = 4;
    public static final int DEFAULT_RUNS = 1;
    public static final String DEFAULT_FILE = "MNIST_Digits/t10kLabeledPoints.txt";

    public void testKMeans() {
        runKMeans(DEFAULT_FILE, DEFAULT_K, DEFAULT_ITERATIONS, DEFAULT_RUNS);
    }

    public static void runKMeans(String inputFile, int k, int iterations, int runs) {
        SparkUtilities.setAppName("KMeans");
        JavaSparkContext sc = SparkUtilities.getCurrentContext();
        JavaRDD<String> lines = sc.textFile(inputFile);

        JavaRDD<LabeledPoint> labeledPoints = lines.map(new LabeledPointUtilities.ParseLabeledPoint());

  //      labeledPoints = labeledPoints.map(new GaborWaveletFeature(MINST_IMAGE_WIDTH,MINST_IMAGE_HEIGHT) );
        labeledPoints = labeledPoints.map(new HoughTransform(MINST_IMAGE_WIDTH,MINST_IMAGE_HEIGHT) );

        labeledPoints = labeledPoints.persist(StorageLevel.MEMORY_AND_DISK());

        JavaRDD<Vector> points = labeledPoints.flatMap(new LabeledPointUtilities.ExtractFeatures());


//        List<Vector> collect1 = points.collect();
//        points = sc.parallelize(collect1);

        KMeansModel model = KMeans.train(points.rdd(), k, iterations, runs, KMeans.K_MEANS_PARALLEL());

        JavaRDD<Classification> classified = labeledPoints.map(new ParsePoint3(model));




        List<Classification> collect = classified.collect();

        int[] countClassificication = new int[10];
        int[][] crossCounts = new int[10][10];


        for (Classification c : collect) {
            crossCounts[c.original_prediction] [c.prediction] ++;
        }

        int[] best_fit = new int[10];
        int[] best_class = new int[10];

        for (int i = 0; i < crossCounts.length; i++) {

            int[] crossCount = crossCounts[i];
            for (int j = 0; j < crossCount.length; j++) {
                int i1 = crossCount[j];
                if(i1 > best_fit[i]) {
                    best_fit[i] = i1;
                    best_class[i] = j;
                }
            }
        }

        for (int i = 0; i < best_class.length; i++) {
            int best_clas = best_class[i];
            int fit = best_fit[i];
            System.out.println("index " + i + " " + best_clas + " " + fit);

        }



//        System.out.println("Cluster centers:");
//        for (Vector center : model.clusterCenters()) {
//            System.out.println(" " + center);
//        }
        double cost = model.computeCost(points.rdd());
        System.out.println("Cost: " + cost);

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

        runKMeans(inputFile, k, iterations, runs);

        //sc.stop();
    }
}