package com.lordjoe.machine_learning.tictactoe;

import com.lordjoe.distributed.SparkUtilities;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * com.lordjoe.machine_learning.tictactoe.NeuralNetTrainingSet
 * utilities
 * User: Steve
 * Date: 4/7/2016
 */
public class NeuralNetTrainingSet {

    public static final int INITIAL_SIZE = 1000;
    public static final int INCREMENT_SIZE = 1000 * 1000;
    public static final Integer ONE = new Integer(1);

    /**
     * use flatmap to multiply an RDD size by INCREMENT_SIZE create an RDD of Integer 1
     * them Map will make the real RDD
     */
    public static class MultiplyByIncrement implements FlatMapFunction<Integer, Integer>, Serializable {
        private final int increment;

        public MultiplyByIncrement(int increment) {
            this.increment = increment;
        }

        @Override
        public Iterable<Integer> call(Integer integer) throws Exception {
            List<Integer> holder = new ArrayList<Integer>();
            for (int i = 0; i < increment; i++) {
                holder.add(ONE);
            }
            return holder;
        }
    }

    public static final Integer[] INITIAL_ARRAY = new Integer[1000];

    /**
     * produce an RDD of integers of at least the requested size -
     * all holding One = this allows a Mapping function to fill in
     * values for training
     *
     * @param ctx
     * @param size
     * @return
     */
    public static JavaRDD<Integer> getRDDOfSize(JavaSparkContext ctx, long psize) {
        Arrays.fill(INITIAL_ARRAY, ONE);
        JavaRDD<Integer> ret = ctx.parallelize(Arrays.asList(INITIAL_ARRAY));
        long size = psize;
        size  = 1 + size / INITIAL_SIZE;
          while (size > 1) {
            if (size < INCREMENT_SIZE) {
                ret = ret.flatMap(new MultiplyByIncrement((int) size));
                size = 0;
            } else {
                ret = ret.flatMap(new MultiplyByIncrement(INCREMENT_SIZE));
                size = 1 + size / INCREMENT_SIZE;
            }
        }
        return ret;
    }

    public static JavaRDD<LabeledPoint>  simulateGames(int numberGames, IInputMapping mapping,IPlayer x,IPlayer y)
    {
        JavaRDD<Integer> items = getRDDOfSize(SparkUtilities.getCurrentContext(),numberGames);
        JavaRDD<LabeledPoint> ret = items.flatMap(new GamePlayer(x,y,mapping));
        return  ret;
    }

    public static class GamePlayer implements FlatMapFunction<Integer,LabeledPoint>, Serializable {

        private final IPlayer x;
        private final IPlayer o;
        private final IInputMapping mapping;

        public GamePlayer(IPlayer x, IPlayer o, IInputMapping mapping) {
            this.x = x;
            this.o = o;
            this.mapping = mapping;
        }

        @Override
        public Iterable<LabeledPoint> call(Integer integer) throws Exception {
            List<LabeledPoint> holder = new ArrayList<LabeledPoint>();
            Game g = new Game(x,o);
            List<Board> playBoards = g.getPlayBoards();
            Player winner = g.getWinner();
            double score = 0;
            if(winner != null)
                score += winner.value;
            double numberMoves = playBoards.size();
            int move = 1;
            for (Board playBoard : playBoards) {
                  double myScore = 1 + ((move++) * score / numberMoves );
                  holder.add(new LabeledPoint(myScore,mapping.boardToVector(playBoard)));
            }

            return holder;
           }
    }

}
