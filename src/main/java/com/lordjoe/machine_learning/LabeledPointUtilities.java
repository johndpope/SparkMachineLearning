package com.lordjoe.machine_learning;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import javax.annotation.Nonnull;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * com.lordjoe.machine_learning.LabeledPointUtilities
 * User: Steve
 * Date: 3/15/2016
 */
public class LabeledPointUtilities {
    private static final Pattern SPACE = Pattern.compile(" ");

    public static class ParseLabeledPoint implements Function<String, LabeledPoint>, Serializable  {
        private static final Pattern SPACE = Pattern.compile(" ");

        @Override
        public LabeledPoint call(String line) {
            LabeledPoint parse = LabeledPoint.parse(line);
            return parse;
        }
    }

    /**
     * function to return the Features from a LabeledPoint
     */
    public static class ExtractFeatures implements FlatMapFunction<LabeledPoint, Vector>, Serializable {

        @Override
        public Iterable<Vector> call(LabeledPoint pt) {
            List<Vector>  ret = new ArrayList<>( );
            Vector features = pt.features();
            if(features.size() > 3)
                ret.add(features) ;
            return ret;
        }
    }

    /**
     *
     * @param s
     * @return
     */
    public static LabeledPoint labeledPointFromString(String s) {
        return LabeledPoint.parse(s);
    }

    /**
     * function to test equivalence of LabeledPoints
     * @param p1  point 1
     * @param p2  point2
     * @param epsilon positive allowed difference
     * @return  true if they are the same within limits
     */
    public static boolean equivalent(@Nonnull  LabeledPoint p1,@Nonnull LabeledPoint p2, double epsilon)
    {
        if(Math.abs(p1.label() - p2.label()) > epsilon)
            return false;
        Vector v1 = p1.features();
        Vector v2 = p2.features();
        return equivalent(v1,v2,epsilon);
    }


    /**
     * function to test equivalence of LabeledPoints
     * @param v1  vector 1
     * @param v2  vector2
     * @param epsilon positive allowed difference
     * @return  true if they are the same within limits
     */
    public static boolean equivalent(Vector v1,Vector v2,double epsilon)
    {
        if(v1.size() != v2.size())
            return false;
        double[] d1 = v1.toArray();
        double[] d2 = v2.toArray();
        for (int i = 0; i < d2.length; i++) {
            double vp1 = d1[i];
             double vp2 = d2[i];
            if(  Math.abs(vp1 - vp2)  > epsilon)
               return false;
           }
        return true;
    }

    /**
     * convert an array of doubles to a sparse vector
     * @param values
     * @return
     */
    public static Vector toSparseVector(double[] values)
    {
        List<Integer> indexes = new ArrayList<>();
        List<Double> vals= new ArrayList<>();

        /**
         * makle a list of non-zero values
         */
        int vectorSize = values.length;
        for (int p = 0; p < vectorSize; p++) {
            double v = values[p];
              if (v != 0) {
                indexes.add(p);
                vals.add(v);
            }
        }

        int[] indx = new int[indexes.size()];
        double[] vlx = new double[indexes.size()];
        for (int j = 0; j < vals.size(); j++) {
            indx[j] = indexes.get(j);
            vlx[j] = vals.get(j);

        }

        Vector sparse = Vectors.sparse(vectorSize, indx, vlx);
        return sparse;
    }

    /**
     * convert an array of doubles to a sparse vector
     * @param values
     * @return
     */
    public static Vector toSparseVector2(double[] values)
    {
        Vector dense = Vectors.dense(values);
        Vector sparse = dense.toSparse();
        return sparse;
    }

    /**
     * merge values of several vectors
     * @param points
     * @return
     */
    public static Vector combine(Vector... points) {
        List<Double> holder = new ArrayList<>();
        for (int i = 0; i < points.length; i++) {
            Vector point = points[i];
            for (int j = 0; j < point.size(); j++) {
                 holder.add(point.apply(j)) ;
            }
        }
        double[] values = new double[holder.size()];
        for (int i = 0; i < values.length; i++) {
             values[i] = holder.get(i);
         }

        return Vectors.dense(values) ;
    }

    /**
     * merge the features of several points having the same label
     * @param p1
     * @param points
     * @return
     */
    public static LabeledPoint combine(LabeledPoint p1,LabeledPoint... points) {
        Vector[] features = new Vector[points.length + 1];
        double label = p1.label();
        features[0] = p1.features();
        for (int i = 0; i < points.length; i++) {
            LabeledPoint point = points[i];
            if(Math.abs(label - point.label()) > 0.001)
                throw new IllegalStateException("combined points must have the same label");
            features[i + 1] = point.features();
        }
        return new LabeledPoint(label,combine(features));
    }

    /**
     * display metrics for a run on test data
     * @param predictionAndLabels
     */
    public static void showMulticlassMetrics(JavaRDD<Tuple2<Object, Object>> predictionAndLabels) {
        // Get evaluation metrics.
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());

        // Confusion matrix
        Matrix confusion = metrics.confusionMatrix();
        System.out.println("Confusion matrix: \n" + confusion);

        // Overall statistics
        System.out.println("Precision = " + metrics.precision());
        System.out.println("Recall = " + metrics.recall());
        System.out.println("F1 Score = " + metrics.fMeasure());

        // Stats by labels
        for (int i = 0; i < metrics.labels().length; i++) {
            System.out.format("Class %f precision = %f\n", metrics.labels()[i],metrics.precision(
                    metrics.labels()[i]));
            System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
                    metrics.labels()[i]));
            System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
                    metrics.labels()[i]));
        }

        //Weighted stats
        System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
        System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
        System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
        System.out.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());
    }

}
