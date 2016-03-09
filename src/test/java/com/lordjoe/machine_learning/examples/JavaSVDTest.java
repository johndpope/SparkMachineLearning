/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.lordjoe.machine_learning.examples;

// $example on$
import java.util.LinkedList;
// $example off$

import com.lordjoe.distributed.SparkUtilities;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
// $example on$
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.junit.Test;
import scala.Option;
// $example off$

/**
 * Example for SingularValueDecomposition.
 */
public class JavaSVDTest {


  @Test
  public void testSVD() {
    SparkUtilities.setAppName("SVD Example");
    JavaSparkContext sc = SparkUtilities.getCurrentContext();

       // $example on$
    double[][] array = {{1.12, 2.05, 3.12}, {5.56, 6.28, 8.94}, {10.2, 8.0, 20.5}};
    LinkedList<Vector> rowsList = new LinkedList<Vector>();
    for (int i = 0; i < array.length; i++) {
      Vector currentRow = Vectors.dense(array[i]);
      rowsList.add(currentRow);
    }
    JavaRDD<Vector> rows =  sc.parallelize(rowsList);

    // Create a RowMatrix from JavaRDD<Vector>.
    RowMatrix mat = new RowMatrix(rows.rdd());

    // Compute the top 3 singular values and corresponding singular vectors.
    SingularValueDecomposition<RowMatrix, Matrix> svd = mat.computeSVD(3, true, 1.0E-9d);
    RowMatrix U = svd.U();
    Vector s = svd.s();
    Matrix V = svd.V();
    // $example off$
    Vector[] collectPartitions = (Vector[]) U.rows().collect();
    System.out.println("U factor is:");
    for (Vector vector : collectPartitions) {
      System.out.println("\t" + vector);
    }
    System.out.println("Singular values are: " + s);
    System.out.println("V factor is:\n" + V);
  }
}