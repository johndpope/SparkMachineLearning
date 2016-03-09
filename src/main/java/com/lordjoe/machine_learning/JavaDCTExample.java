package com.lordjoe.machine_learning;

/**
 * com.lordjoe.machine_learning.JavaDCTExample
 * User: Steve
 * Date: 3/1/2016
 */

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SQLContext;

// $example on$
import java.util.Arrays;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.DCT;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Option;
// $example off$

public class JavaDCTExample {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("JavaDCTExample");
        Option<String> option = conf.getOption("spark.master");
        if (!option.isDefined())    // use local over nothing
            conf.setMaster("local[*]");

        JavaSparkContext jsc = new JavaSparkContext(conf);
        SQLContext jsql = new SQLContext(jsc);

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
        // $example off$
        jsc.stop();
    }
}

