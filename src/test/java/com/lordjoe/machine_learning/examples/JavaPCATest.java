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
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.junit.Test;
import scala.Option;
// $example off$

/**
 * Example for compute principal components on a 'RowMatrix'.
 */
public class JavaPCATest {
  @Test
  public void testLogisticRegressionWithSGD() {
    SparkUtilities.setAppName("PCA Example");
    JavaSparkContext  jsc = SparkUtilities.getCurrentContext();

    // $example on$
    double[][] array = {{1.12, 2.05, 3.12}, {5.56, 6.28, 8.94}, {10.2, 8.0, 20.5}};
    LinkedList<Vector> rowsList = new LinkedList<Vector>();
    for (int i = 0; i < array.length; i++) {
      Vector currentRow = Vectors.dense(array[i]);
      rowsList.add(currentRow);
    }
    JavaRDD<Vector> rows = jsc.parallelize(rowsList);

    // Create a RowMatrix from JavaRDD<Vector>.
    RowMatrix mat = new RowMatrix(rows.rdd());

    // Compute the top 3 principal components.
    Matrix pc = mat.computePrincipalComponents(3);
    RowMatrix projected = mat.multiply(pc);
    // $example off$
    Vector[] collectPartitions = (Vector[])projected.rows().collect();
    System.out.println("Projected vector of principal component:");
    for (Vector vector : collectPartitions) {
      System.out.println("\t" + vector);
    }
  }
}
