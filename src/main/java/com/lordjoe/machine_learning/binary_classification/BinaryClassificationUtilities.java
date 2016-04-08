package com.lordjoe.machine_learning.binary_classification;

import com.lordjoe.machine_learning.LabeledPointUtilities;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;
import scala.tools.cmd.gen.AnyVals;

import java.io.Serializable;

/**
 * com.lordjoe.machine_learning.binary_classification.BinaryClassificationUtilities
 * User: Steve
 * Date: 3/23/2016
 */
public class BinaryClassificationUtilities {

    /**
     * given 2 labels - say 5.0 and 8.0 map them into 0.0 and 1.0 -
     * this converts a multiple classification into a binary one
     */
    public static class RemapLabels implements Function<LabeledPoint, LabeledPoint>,Serializable {
        private final double[] oldLabels;

        public RemapLabels(double[] oldLabels) {
            if(oldLabels.length != 2)
                throw new IllegalArgumentException("mustt pass 2 doubles");
            this.oldLabels = oldLabels;
        }

        @Override
        public LabeledPoint call(LabeledPoint v1) throws Exception {
            double newLabel = 0;
            if(Math.abs(v1.label() - oldLabels[1])    < 0.001)
                newLabel = 1.0;
            else {
                if(Math.abs(v1.label() - oldLabels[0])    > 0.001)
                    throw new IllegalStateException("label must be " + oldLabels[0] + " or " +  oldLabels[1] + " not " + v1.label()); // ToDo change
            }
            return new LabeledPoint(newLabel,v1.features());

        }
    }

    /**
     * given 2 labels - say 5.0 and 8.0 map them into 0.0 and 1.0 -
     * this converts a multiple classification into a binary one
     */
    public static JavaRDD<LabeledPoint> remapLabels(JavaRDD<LabeledPoint> inp,double[] mapping) {
        return inp.map(new RemapLabels(mapping));
    }

    public static class DoubleFromString implements Function<Tuple2<Object, Object>, Double>,Serializable {
        @Override
        public Double call(Tuple2<Object, Object> t) {
            return new Double(t._1().toString());
        }
    }


    public static void showAllStatistics(JavaRDD<Tuple2<Object, Object>> predictionAndLabels) {
        predictionAndLabels = predictionAndLabels.persist(StorageLevel.MEMORY_AND_DISK());

        BinaryClassificationUtilities.showStatistics(predictionAndLabels);
        System.out.println("===========================");
        LabeledPointUtilities.showMulticlassMetrics(predictionAndLabels);
    }


    public static void showStatistics(JavaRDD<Tuple2<Object, Object>> predictionAndLabels) {
        // Get evaluation metrics.
        BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(predictionAndLabels.rdd());

        // Precision by threshold
        JavaRDD<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD();
        System.out.println("Precision by threshold: " + precision.collect());

        // Recall by threshold
        JavaRDD<Tuple2<Object, Object>> recall = metrics.recallByThreshold().toJavaRDD();
        System.out.println("Recall by threshold: " + recall.collect());

        // F Score by threshold
        JavaRDD<Tuple2<Object, Object>> f1Score = metrics.fMeasureByThreshold().toJavaRDD();
        System.out.println("F1 Score by threshold: " + f1Score.collect());

        JavaRDD<Tuple2<Object, Object>> f2Score = metrics.fMeasureByThreshold(2.0).toJavaRDD();
        System.out.println("F2 Score by threshold: " + f2Score.collect());

        // Precision-recall curve
        JavaRDD<Tuple2<Object, Object>> prc = metrics.pr().toJavaRDD();
        System.out.println("Precision-recall curve: " + prc.collect());

        // Thresholds
        JavaRDD<Double> thresholds = precision.map(
                new DoubleFromString()
        );

        // ROC Curve
        JavaRDD<Tuple2<Object, Object>> roc = metrics.roc().toJavaRDD();
        System.out.println("ROC curve: " + roc.collect());

        // AUPRC
        System.out.println("Area under precision-recall curve = " + metrics.areaUnderPR());

        // AUROC
        System.out.println("Area under ROC = " + metrics.areaUnderROC());
    }
}
