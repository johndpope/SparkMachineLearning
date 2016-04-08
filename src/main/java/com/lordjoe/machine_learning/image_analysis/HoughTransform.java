package com.lordjoe.machine_learning.image_analysis;

import com.lordjoe.machine_learning.LabeledPointUtilities;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.io.Serializable;

/**
 * combinwe line and circle Hough transform
 * com.lordjoe.machine_learning.image_analysis.HoughTransform
 * User: Steve
 * Date: 3/22/2016
 */
public class HoughTransform implements Function<LabeledPoint, LabeledPoint>, Serializable {
    // the width and height of the image
    protected final int width, height;
    private final HoughLineTransform lt;
    private final HoughCircles ct;

    public HoughTransform(int width, int height) {
        this.width = width;
        this.height = height;
        lt = new HoughLineTransform(width,height);
        ct = new HoughCircles(width,height);
    }

    @Override
    public LabeledPoint call(LabeledPoint v1) throws Exception {
        LabeledPoint p1 = lt.call(v1);
        LabeledPoint p2 = ct.call(v1);
        return LabeledPointUtilities.combine(p1,p2);
    }
}
