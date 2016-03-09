package com.lordjoe.distributed;



import org.apache.spark.api.java.function.Function;

import java.io.Serializable;

/**
 * com.lordjoe.distributed.SerializableFunction
 * User: Steve
 * Date: 3/8/2016
 */
public interface SerializableFunction<T,R> extends Function<T,R>,Serializable {
  }
