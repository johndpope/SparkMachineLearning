package com.lordjoe.machine_learning.image_analysis;

/**
 * com.lordjoe.machine_learning.image_analysis.GaborWaveletFeature
 * User: Steve
 * Date: 3/21/2016
 */
/*
 * This file is part of the LIRE project: http://lire-project.net
 * LIRE is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LIRE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with LIRE; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * We kindly ask you to refer the any or one of the following publications in
 * any publication mentioning or employing Lire:
 *
 * Lux Mathias, Savvas A. Chatzichristofis. Lire: Lucene Image Retrieval –
 * An Extensible Java CBIR Library. In proceedings of the 16th ACM International
 * Conference on Multimedia, pp. 1085-1088, Vancouver, Canada, 2008
 * URL: http://doi.acm.org/10.1145/1459359.1459577
 *
 * Lux Mathias. Content Based Image Retrieval with LIRE. In proceedings of the
 * 19th ACM International Conference on Multimedia, pp. 735-738, Scottsdale,
 * Arizona, USA, 2011
 * URL: http://dl.acm.org/citation.cfm?id=2072432
 *
 * Mathias Lux, Oge Marques. Visual Information Retrieval using Java and LIRE
 * Morgan & Claypool, 2013
 * URL: http://www.morganclaypool.com/doi/abs/10.2200/S00468ED1V01Y201301ICR025
 *
 * Copyright statement:
 * ====================
 * (c) 2002-2013 by Mathias Lux (mathias@juggle.at)
 *  http://www.semanticmetadata.net/lire, http://www.lire-project.net
 *
 * Updated: 11.07.13 10:31
 */
//package net.semanticmetadata.lire.imageanalysis.features.global;


import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import javax.annotation.Nonnull;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.BufferedImageOp;
import java.awt.image.ColorConvertOp;
import java.awt.image.Raster;
import java.io.Serializable;

/**
 * Implementation of a Gabor texture features done by  Marko Keuschnig & Christian Penz<br>
 * Fixed 2011-05-10 based on the comments of Arthur Lin.
 */

public class GaborWaveletFeature  implements Function<LabeledPoint, LabeledPoint>, Serializable {


    private static final double U_H = .4;
    private static final double U_L = .05;
    private static final int S = 4, T = 4; // filter mask size
 //   private static final int M = 5, N = 6; // scale & orientation
    private static final int M = 4, N = 4; // scale & orientation

    private static final int MAX_IMG_HEIGHT = 64;

    private static final double A = Math.pow((U_H / U_L), 1. / (M - 1));
    private static double[] theta = new double[N];
    private static double[] modulationFrequency = new double[M];
    private static double[] sigma_x = new double[M];
    private static double[] sigma_y = new double[M];
    private static transient double[][][][][] selfSimilarGaborWavelets = null;

    private static final double LOG2 = Math.log(2);

    private final int widthInPixels;
    private final int heightInPixels;
    private  double[][][][][] gaborWavelet = null;

    public GaborWaveletFeature(int widthInPixels,int heightInPixels ) {
        this.widthInPixels = widthInPixels;
        this.heightInPixels = heightInPixels;
        guaranteeSelfSimilarGaborWavelets();
    }

    private double[][][][][]  precomputeGaborWavelet( int[][] image ) {
        int length = widthInPixels * heightInPixels;
        double[][][][][] gw = new double[length - S][widthInPixels - T][M][N][2];
        double[] gaborWavelet;
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                for (int x = S; x <  length; x++) {
                    for (int y = T; y < widthInPixels; y++) {
                        gaborWavelet = gaborWavelet(image, x, y, m, n);
                        gw[x - S][y - T][m][n][0] = gaborWavelet[0];
                        gw[x - S][y - T][m][n][1] = gaborWavelet[1];
                    }
                }
            }
        }
        return gw;
    }

    private static synchronized void guaranteeSelfSimilarGaborWavelets() {
        if (selfSimilarGaborWavelets != null)
            return;
        selfSimilarGaborWavelets = new double[S][T][M][N][2];

        for (int i = 0; i < N; i++) {
            theta[i] = i * Math.PI / N;
        }
        for (int i = 0; i < M; i++) {
            modulationFrequency[i] = Math.pow(A, i) * U_L;
            sigma_x[i] =
                    (A + 1) * Math.sqrt(2 * LOG2) /
                            (2 * Math.PI * Math.pow(A, i) * (A - 1) * U_L);
            sigma_y[i] = 1 / (2 * Math.PI * Math.tan(Math.PI / (2 * N)) * Math.sqrt(Math.pow(U_H, 2) / (2 * LOG2) - Math.pow(1 / (2 * Math.PI * sigma_x[i]), 2)));

        }
        double[] selfSimilarGaborWavelet;
        for (int s = 0; s < S; s++) {
            for (int t = 0; t < T; t++) {
                for (int m = 0; m < M; m++) {
                    for (int n = 0; n < N; n++) {
                        selfSimilarGaborWavelet = selfSimilarGaborWavelet(s, t, m, n);
                        selfSimilarGaborWavelets[s][t][m][n][0] = selfSimilarGaborWavelet[0];
                        selfSimilarGaborWavelets[s][t][m][n][1] = selfSimilarGaborWavelet[1];
                    }
                }
            }
        }

    }


    public static double getDistance(double[] targetFeatureVector, double[] queryFeatureVector) {
        double distance = 0;
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                distance += Math.sqrt(Math.pow(queryFeatureVector[m * 2 * N + n * 2] - targetFeatureVector[m * 2 * N + n * 2], 2) + Math.pow(queryFeatureVector[m * 2 * N + n * 2 + 1] - targetFeatureVector[m * 2 * N + n * 2 + 1], 2));
            }
        }

        return distance;
    }


    public static double[] normalize(double[] featureVector) {
        int dominantOrientation = 0;
        double orientationVectorSum = 0;
        double orientationVectorSum2 = 0;
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                orientationVectorSum2 += Math.sqrt(Math.pow(featureVector[m * 2 * N + n * 2], 2) + Math.pow(featureVector[m * 2 * N + n * 2 + 1], 2));
            }
            if (orientationVectorSum2 > orientationVectorSum) {
                orientationVectorSum = orientationVectorSum2;
                dominantOrientation = m;
            }
        }

        double[] normalizedFeatureVector = new double[featureVector.length];
        for (int m = dominantOrientation, k = 0; m < M; m++, k++) {
            for (int n = 0; n < N; n++) {
                normalizedFeatureVector[k * 2 * N + n * 2] = featureVector[m * 2 * N + n * 2];
                normalizedFeatureVector[k * 2 * N + n * 2 + 1] = featureVector[m * 2 * N + n * 2 + 1];
            }
        }
        for (int m = 0, k = M - dominantOrientation; m < dominantOrientation; m++, k++) {
            for (int n = 0; n < N; n++) {
                normalizedFeatureVector[k * 2 * N + n * 2] = featureVector[m * 2 * N + n * 2];
                normalizedFeatureVector[k * 2 * N + n * 2 + 1] = featureVector[m * 2 * N + n * 2 + 1];
            }
        }

        return normalizedFeatureVector;
    }

    @Override
    public LabeledPoint call(LabeledPoint image) {
        return getFeature(image,widthInPixels );
      }

    /**
     * convert a labeled point where the features are integer grey scales 0..255
     * to a Gabor wavelet feature
     * @param pt
     * @param widthInPixels width of the image
     * @return
     */
    public   LabeledPoint getFeature(@Nonnull  LabeledPoint pt, int widthInPixels) {
        double[] gabor = getFeature(pt.features(), widthInPixels);
        Vector dense = Vectors.dense(gabor);
        LabeledPoint ret = LabeledPoint.apply(pt.label(), dense);
        return ret;
    }

    public   double[] getFeature(Vector points, int widthInPixels) {

        int heightInPixels = points.size() / widthInPixels;
        int[][] grayLevel = new int[widthInPixels][heightInPixels];

        gaborWavelet = precomputeGaborWavelet(grayLevel);
        int[] tmp = new int[3];
        for (int i = 0; i < widthInPixels; i++) {
            int index = heightInPixels * i;

            for (int j = 0; j < heightInPixels; j++) {
                grayLevel[i][j] = (int) points.apply(index + j);
            }
        }

        double[] featureVector = new double[M * N * 2];
        double[][] magnitudes = computeMagnitudes(grayLevel);
        int imageSize = points.size();
        double[][] magnitudesForVariance = new double[M][N];


        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                featureVector[m * 2 * N + n * 2] = magnitudes[m][n] / imageSize;
                for (int i = 0; i < magnitudesForVariance.length; i++) {
                    for (int j = 0; j < magnitudesForVariance[0].length; j++) {
                        magnitudesForVariance[i][j] = 0.;
                    }
                }
                for (int x = S; x < widthInPixels; x++) {
                    for (int y = T; y < heightInPixels; y++) {
                        magnitudesForVariance[m][n] += Math.pow(Math.sqrt(Math.pow(this.gaborWavelet[x - S][y - T][m][n][0], 2) + Math.pow(this.gaborWavelet[x - S][y - T][m][n][1], 2)) - featureVector[m * 2 * N + n * 2], 2);
                    }
                }

                featureVector[m * 2 * N + n * 2 + 1] = Math.sqrt(magnitudesForVariance[m][n]) / imageSize;
            }
        }
        gaborWavelet = null;
        return featureVector;
    }


    private double[][] computeMagnitudes(int[][] image) {
        double[][] magnitudes = new double[M][N];
        for (int i = 0; i < magnitudes.length; i++) {
            for (int j = 0; j < magnitudes[0].length; j++) {
                magnitudes[i][j] = 0.;
            }
        }

        if (this.gaborWavelet == null) {
            precomputeGaborWavelet(image);
        }

        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                for (int x = S; x < image.length; x++) {
                    for (int y = T; y < image[0].length; y++) {
                        magnitudes[m][n] += Math.sqrt(Math.pow(this.gaborWavelet[x - S][y - T][m][n][0], 2) + Math.pow(this.gaborWavelet[x - S][y - T][m][n][1], 2));

                    }
                }
            }
        }
        return magnitudes;
    }

    // returns 2 doubles representing the real ([0]) and imaginary ([1]) part of the mother wavelet
    private static double[] gaborWavelet(int[][] img, int x, int y, int m, int n) {
        double re = 0;
        double im = 0;
        for (int s = 0; s < S; s++) {
            for (int t = 0; t < T; t++) {
                try {
                    re += img[x][y] * selfSimilarGaborWavelets[s][t][m][n][0];
                    im += img[x][y] * -selfSimilarGaborWavelets[s][t][m][n][1];
                } catch (Exception e) {
                    throw new RuntimeException(e);

                }
            }
        }

        return new double[]{re, im};
    }

    // returns 2 doubles representing the real ([0]) and imaginary ([1]) part of the mother wavelet
    private static double[] computeMotherWavelet(double x, double y, int m, int n) {

        return new double[]{
                1 / (2 * Math.PI * sigma_x[m] * sigma_y[m]) *
                        Math.exp(-1 / 2 * (Math.pow(x, 2) / Math.pow(sigma_x[m], 2) + Math.pow(y, 2) / Math.pow(sigma_y[m], 2))) *
                        Math.cos(2 * Math.PI * modulationFrequency[m] * x),
                1 / (2 * Math.PI * sigma_x[m] * sigma_y[m]) *
                        Math.exp(-1 / 2 * (Math.pow(x, 2) / Math.pow(sigma_x[m], 2) + Math.pow(y, 2) / Math.pow(sigma_y[m], 2))) *
                        Math.sin(2 * Math.PI * modulationFrequency[m] * x)};
    }

    private static double x_tilde(int x, int y, int m, int n) {
        return
                Math.pow(A, -m) * (x * Math.cos(theta[n]) + y * Math.sin(theta[n]));
    }

    private static double y_tilde(int x, int y, int m, int n) {
        return
                Math.pow(A, -m) * (-x * Math.sin(theta[n] + y * Math.cos(theta[n])));
    }

    private static double[] selfSimilarGaborWavelet(int x, int y, int m, int n) {
        double[] motherWavelet = computeMotherWavelet(x_tilde(x, y, m, n), y_tilde(x, y, m, n), m, n);
        return new double[]{
                Math.pow(A, -m) * motherWavelet[0],
                Math.pow(A, -m) * motherWavelet[1]};
    }


}
