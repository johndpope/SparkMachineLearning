package com.lordjoe.machine_learning.tictactoe;


import com.lordjoe.algorithms.Long_Formatter;
import com.lordjoe.distributed.SparkUtilities;
import com.sun.org.apache.xml.internal.security.signature.reference.ReferenceNodeSetData;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

/**
 * com.lordjoe.machine_learning.tictactoe.NeuralTrainingSetTest
 * User: Steve
 * Date: 4/7/2016
 */
public class NeuralTrainingSetTest {
    public static final NeuralTrainingSetTest[] EMPTY_ARRAY = {};

    public static final Random RND = new Random();
    public static final long ONE_THOUSAND = 1000L;
    public static final long MAX_TEST_DATA = ONE_THOUSAND * ONE_THOUSAND * 2;

    public static long makeTestLong() {
        double lg = Math.log10(MAX_TEST_DATA);
        double val = RND.nextDouble() * lg;
        double realV = Math.pow(val, 10);
        return (long) realV + ONE_THOUSAND; // add ONE_THOUSAND to drop off smaller cases
    }

    /**
     * this takes a long time so it is usually commented out
     * @throws Exception
     */
    @Test
    public void testGetRDDofSizeHuge() throws Exception {
        String neuralTrainingSetTest = "NeuralTrainingSetTest";
        SparkUtilities.setAppName(neuralTrainingSetTest);
        JavaSparkContext sc = SparkUtilities.getCurrentContext();

        long test =  ONE_THOUSAND * ONE_THOUSAND * ONE_THOUSAND * 20; // 20 billion
         JavaRDD<Integer> rddOfSize = NeuralNetTrainingSet.getRDDOfSize(sc, test);
        long count = rddOfSize.count();
        if (count < test) {
             Assert.assertTrue(count >= test);
        }
        Assert.assertTrue(count <= NeuralNetTrainingSet.INCREMENT_SIZE * test);
        System.out.println("desired " + Long_Formatter.format(test) + " created " + Long_Formatter.format(count));
    }

    /**
     * test the ability to create an RDD at least as big as requested
     * and not too much over - used to make random training RDDs
     * @throws Exception
     */
    @Test
    public void testGetRDDofSize() throws Exception {
        String neuralTrainingSetTest = "NeuralTrainingSetTest";
        SparkUtilities.setAppName(neuralTrainingSetTest);
        JavaSparkContext sc = SparkUtilities.getCurrentContext();

        for (int i = 0; i < 20; i++) {
            long test = makeTestLong();
            System.out.println("Test Size " + Long_Formatter.format(test));
            JavaRDD<Integer> rddOfSize = NeuralNetTrainingSet.getRDDOfSize(sc, test);
            long count = rddOfSize.count();
            if (count < test) {
                System.out.println("desired " + Long_Formatter.format(test) + " created " + Long_Formatter.format(count));
                Assert.assertTrue(count >= test);
            }
            Assert.assertTrue(count <= NeuralNetTrainingSet.INCREMENT_SIZE * test);
        }
    }
}
