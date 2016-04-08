package com.lordjoe.algorithms;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

import com.lordjoe.machine_learning.LabeledPointUtilities;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

/**
 * com.lordjoe.algorithms.IDXReader
 * User: Steve
 * Date: 3/14/2016
 */
public class NistImagesToLabeledPoints {
    /**
     * convert NIST sample images to feature vectors of 400 point values
     *
     * @param inImage
     * @param inLabel
     * @param out
     */
    public static void saveImagesAsLabeledPoints(FileInputStream inImage, FileInputStream inLabel, PrintWriter out) {
        int[] hashMap = new int[10];

        try {

            int magicNumberImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfRows = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfColumns = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());

            int magicNumberLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());
            int numberOfLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());

            BufferedImage image = new BufferedImage(numberOfColumns, numberOfRows, BufferedImage.TYPE_INT_ARGB);
            int numberOfPixels = numberOfRows * numberOfColumns;
            double[] featurePixels = new double[numberOfPixels];


            for (int i = 0; i < numberOfImages; i++) {
                double label = (double) inLabel.read();
                List<Integer> indexes = new ArrayList<>();
                List<Integer> values = new ArrayList<>();


                if (i % 100 == 0) {
                    System.out.println("Number of images extracted: " + i);
                }

                for (int p = 0; p < numberOfPixels; p++) {
                    int gray = inImage.read();
                    featurePixels[p] = gray;
                    if (gray != 0) {
                        indexes.add(p);
                        values.add(gray);
                    }
                }

                int[] indx = new int[indexes.size()];
                double[] vals = new double[indexes.size()];
                for (int j = 0; j < vals.length; j++) {
                    indx[j] = indexes.get(j);
                    vals[j] = values.get(j);

                }

                Vector sparse = Vectors.sparse(numberOfPixels, indx, vals);
                Vector dense = Vectors.dense(featurePixels);

                if (!LabeledPointUtilities.equivalent(sparse, dense, 0.0001))
                    throw new UnsupportedOperationException("Fix This"); // ToDo
                LabeledPoint pt = new LabeledPoint(label, sparse);

                String outLine = pt.toString();
                out.println(outLine);

            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    public static void convertImages(String imageFileName, String lablelFileName, String outFileName) throws Exception {
        // TODO Auto-generated method stub
        FileInputStream inImage = null;
        FileInputStream inLabel = null;
        PrintWriter out = null;

        String base_dir = new File(".").getAbsolutePath();
        base_dir = base_dir.replace("\\", "/").replace("/.", "/");

        String inputImagePath = base_dir + imageFileName;
        String inputLabelPath = base_dir + lablelFileName;
        String outPath = base_dir + outFileName;

        out = new PrintWriter(new FileWriter(outPath));

        int[] hashMap = new int[10];

        try {
            inImage = new FileInputStream(inputImagePath);
            inLabel = new FileInputStream(inputLabelPath);
            saveImagesAsLabeledPoints(inImage, inLabel, out);

        } catch (IOException e) {
            throw new RuntimeException(e);
        } finally

        {
            if (inImage != null) {
                try {
                    inImage.close();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            if (inLabel != null) {
                try {
                    inLabel.close();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            out.close();
        }
    }

    /**
     * sample args
     * t10k-images.idx3-ubyte t10k-labels.idx1-ubyte t10kLabeledPoints.txt
     * train-images.idx3-ubyte train-labels.idx1-ubyte trainLabeledPoints.txt
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {

//        String base_dir = "E:/Spark Machine Learning/MNIST Digits/";
//        //   String inputImagePath = base_dir + "train-images.idx3-ubyte";
//        //    String inputLabelPath = base_dir + "train-labels.idx1-ubyte";
//        String inputImagePath = base_dir + "t10k-images.idx3-ubyte";
//        String inputLabelPath = base_dir + "t10k-labels.idx1-ubyte";
//        String outPath = base_dir + "t10kLabeledPoints.txt";

        convertImages(args[0], args[1], args[2]);

    }

}