package com.lordjoe.machine_learning.tictactoe;

import com.lordjoe.distributed.SparkUtilities;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.ann.FeedForwardTopology;
import org.apache.spark.ml.ann.TopologyModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

import javax.annotation.Nonnull;

import java.util.List;

import static com.lordjoe.machine_learning.tictactoe.NeuralNetTrainingSet.simulateGames;
import static com.lordjoe.machine_learning.tictactoe.Player.O;
import static com.lordjoe.machine_learning.tictactoe.Player.X;

/**
 * com.lordjoe.machine_learning.tictactoe.NeuralNetPlayer
 * User: Steve
 * Date: 4/7/2016
 */
public class NeuralNetPlayer implements  IPlayer {

    public static final int BLOCK_SIZE = 128;
    public static final int MAX_ITERATIONS = 1000;
    public static final int NUMBER_INPUT_SETS = 10  * 1000;

    public static final int[] layers =  { Board.getTotalBoardSize(), Board.getTotalBoardSize(), 3 } ;

    public Player player;
    public final IInputMapping mapping = new PlusMinusMapping();

    private MultilayerPerceptronClassifier classifier = new MultilayerPerceptronClassifier();
    private MultilayerPerceptronClassificationModel model;
    private TopologyModel topoModel;   // see stackoverflow.com/questions/35962952/how-to-get-classification-probabilities-from-multilayerperceptronclassifier

    public NeuralNetPlayer(Player pplayer) {
        player = pplayer;
        classifier.setLayers(layers);
        classifier.setBlockSize(BLOCK_SIZE);
        classifier.setMaxIter(MAX_ITERATIONS);
     }

    public void setPlayer(Player player) {
        this.player = player;
    }

    public void train()
     {
         SQLContext sqctx = SparkUtilities.getCurrentSQLContext();
         RandomPlayer x = new RandomPlayer(X);
         RandomPlayer y = new RandomPlayer(O);
         JavaRDD<LabeledPoint> data =  simulateGames(NUMBER_INPUT_SETS, mapping, x, y);
         DataFrame train = sqctx.createDataFrame(data, LabeledPoint.class);

         model = classifier.fit(train);
         // see stackoverflow.com/questions/35962952/how-to-get-classification-probabilities-from-multilayerperceptronclassifier
           topoModel = FeedForwardTopology.multiLayerPerceptron(model.layers(), true).getInstance(model.weights());
     }



    @Nonnull
    @Override
    public Player getPlayer() {
        return player;
    }


    @Override
    public Position choseMove(@Nonnull Board b) {
        Position bestMove = null;
        Player player = getPlayer();
        double bestScore = -100;
        List<Position> legalMoves = b.getLegalMoves();
         for (Position move : legalMoves) {
             Board test = new Board(b);
            test.makeMove(player,move);
             Vector features = mapping.boardToVector(test);

             Vector probabilies = topoModel.predict(features);
             // use score as P X - P y
             double predictScore =  player.value * (probabilies.apply(2) -  probabilies.apply(0));  // model.predict(features);

            if(predictScore > bestScore)  {
                bestMove = move;
                bestScore = predictScore;
            }
        }
           return bestMove;
    }

    public static void main(String[] args) {
        int matchGames = 10000;
        RandomPlayer x = new RandomPlayer(X);
        RandomPlayer o = new RandomPlayer(O);

        NeuralNetPlayer nnPlayer = new NeuralNetPlayer(X);
        nnPlayer.train();

        Match match;
        String play;

        match = new Match(matchGames,nnPlayer,o) ;
        play = match.play();
        System.out.println(nnPlayer.getPlayer().toString() + " " + play);

        nnPlayer.setPlayer(O);
        match = new Match(matchGames,x,nnPlayer) ;
        play = match.play();
        System.out.println(nnPlayer.getPlayer().toString() + " " + play);

        match = new Match(matchGames,x,o)  ;
         play = match.play();
        System.out.println("Random " + play);

    }
}
