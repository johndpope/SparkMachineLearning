package com.lordjoe.machine_learning.examples;

import com.lordjoe.distributed.SparkUtilities;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.DCT;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;

import java.util.Arrays;

/**
 * com.lordjoe.machine_learning.JavaDCExampleTest
 * User: Steve
 * Date: 3/8/2016
 */
public class JavaDCExampleTest {

    @Test
    public void testDC() throws Exception {
        SparkUtilities.setAppName("JavaDC");
        JavaSparkContext jsc = SparkUtilities.getCurrentContext();

        SQLContext jsql = SparkUtilities.getCurrentSQLContext();
        // $example on$
        JavaRDD<Row> data = jsc.parallelize(Arrays.asList(
                RowFactory.create(Vectors.dense(0.0, 1.0, -2.0, 3.0)),
                RowFactory.create(Vectors.dense(-1.0, 2.0, 4.0, -7.0)),
                RowFactory.create(Vectors.dense(14.0, -2.0, -5.0, 1.0))
        ));
        StructType schema = new StructType(new StructField[]{
                new StructField("features", new VectorUDT(), false, Metadata.empty()),
        });
        DataFrame df = jsql.createDataFrame(data, schema);
        DCT dct = new DCT()
                .setInputCol("features")
                .setOutputCol("featuresDCT")
                .setInverse(false);
        DataFrame dctDf = dct.transform(df);
        dctDf.select("featuresDCT").show(3);

    }
}
