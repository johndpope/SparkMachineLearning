package com.lordjoe.machine_learning.tictactoe;

import com.lordjoe.distributed.SparkUtilities;
import com.lordjoe.testing.ElapsedTimer;
import org.apache.commons.collections.map.HashedMap;
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
import java.util.Map;

import static com.lordjoe.machine_learning.tictactoe.NeuralNetTrainingSet.simulateGames;
import static com.lordjoe.machine_learning.tictactoe.Player.O;
import static com.lordjoe.machine_learning.tictactoe.Player.X;

/**
 * com.lordjoe.machine_learning.tictactoe.NeuralNetPlayer
 * User: Steve
 * Date: 4/7/2016
 */
public class NeuralNetPlayer implements IStrategy {

    public static final int DEFAULT_BLOCK_SIZE = 128;
    public static final int DEFAULT_MAX_ITERATIONS = 100;
    public static final int NUMBER_INPUT_SETS = 10 * 1000;

    public static final int[] DEFAULT_LAYERS = {TicTacToeBoard.getTotalBoardSize(), TicTacToeBoard.getTotalBoardSize(), 3};

    public Player player;
    public final IInputMapping mapping = new PlusMinusMapping();

    private MultilayerPerceptronClassifier classifier = new MultilayerPerceptronClassifier();
    private MultilayerPerceptronClassificationModel model;
    private TopologyModel topoModel;   // see stackoverflow.com/questions/35962952/how-to-get-classification-probabilities-from-multilayerperceptronclassifier

    public NeuralNetPlayer(Player pplayer) {
        this(pplayer, DEFAULT_LAYERS, DEFAULT_BLOCK_SIZE, DEFAULT_MAX_ITERATIONS);
    }

    public NeuralNetPlayer(Player pplayer, int[] layers, int blocksize, int max_terations) {
        player = pplayer;
        classifier.setLayers(layers);
        classifier.setBlockSize(blocksize);
        classifier.setMaxIter(max_terations);
    }

    public void setPlayer(Player player) {
        this.player = player;
    }

    public void train(int numberInputSets) {
        ElapsedTimer timer = new ElapsedTimer();
        SQLContext sqctx = SparkUtilities.getCurrentSQLContext();
        IStrategy x = new MinaMaxPlayer(X);
        IStrategy y = new MinaMaxPlayer(O);
        JavaRDD<LabeledPoint> data = simulateGames(numberInputSets, 0.3, mapping, x, y);

        timer.showElapsed("Built training set");

        DataFrame train = sqctx.createDataFrame(data, LabeledPoint.class);

        model = classifier.fit(train);
        // see stackoverflow.com/questions/35962952/how-to-get-classification-probabilities-from-multilayerperceptronclassifier
        topoModel = FeedForwardTopology.multiLayerPerceptron(model.layers(), true).getInstance(model.weights());
        timer.showElapsed("Finished training set");
    }


    @Nonnull
    @Override
    public Player getPlayer() {
        return player;
    }


    @Override
    public TicTacToeMove choseMove(@Nonnull TicTatToeGame g) {
        TicTacToeBoard b = g.getBoard();
        Position bestMove = null;
        Player player = getPlayer();
        double bestScore = -100;
        double playerValue = player.value;
        List<Position> legalMoves = b.getLegalMoves();
        Map<Position, Vector> winFeatures = new HashedMap();
        Map<Position, Vector> drawFeatures = new HashedMap();
        for (Position move : legalMoves) {
            TicTacToeBoard test = new TicTacToeBoard(b);
            test.makeMove(new TicTacToeMove(player, move));
            Vector features = mapping.boardToVector(test);
            double label = model.predict(features);

            if (label == playerValue) {
                winFeatures.put(move, features);
            }
            if (label == 0) {
                drawFeatures.put(move, features);
            }
        }

        Map<Position, Vector> use = drawFeatures;
        if (winFeatures.size() > 0)
            use = drawFeatures;

        if (use.size() == 0) {
            bestScore = -100;
            for (Position move : legalMoves) {
                TicTacToeBoard test = new TicTacToeBoard(b);
                test.makeMove(new TicTacToeMove(player, move));
                Vector features = mapping.boardToVector(test);
                Vector probabilies = topoModel.predict(features);
                // use score as P X - P y
                double pXWins = probabilies.apply(2);
                double pOWins = probabilies.apply(0);
                double predictScore = player.value * (pXWins - pOWins);  // model.predict(features);

                if (predictScore > bestScore) {
                    bestMove = move;
                    bestScore = predictScore;
                }
            }
        }
        else {
             bestScore = -100;
            for (Position position : use.keySet()) {
                Vector features = use.get(position);
                Vector probabilies = topoModel.predict(features);
                // use score as P X - P y
                double pXWins = probabilies.apply(2);
                double pOWins = probabilies.apply(0);
                double predictScore = player.value * (pXWins - pOWins);  // model.predict(features);

                if (predictScore > bestScore) {
                    bestMove = position;
                    bestScore = predictScore;
                }
             }
        }


        return new TicTacToeMove(player, bestMove);
    }

    public static void main(String[] args) {
        int matchGames = 100;
        MinaMaxPlayer mmX = new MinaMaxPlayer(X);
        MinaMaxPlayer mmO = new MinaMaxPlayer(O);

        RandomPlayer x = new RandomPlayer(X);
        RandomPlayer o = new RandomPlayer(O);

        NeuralNetPlayer nnPlayer = new NeuralNetPlayer(X);
        nnPlayer.train(NUMBER_INPUT_SETS);


        Match match;
        String play;

        match = new Match(matchGames, nnPlayer, o);
        play = match.play();
        System.out.println("ramdom " + nnPlayer.getPlayer().toString() + " " + play);

        match = new Match(matchGames, nnPlayer, mmO);
        play = match.play();
        System.out.println("Minamax " + nnPlayer.getPlayer().toString() + " " + play);

        nnPlayer.setPlayer(O);
        match = new Match(matchGames, mmX, nnPlayer);
        play = match.play();
        System.out.println("Minamax " + nnPlayer.getPlayer().toString() + " " + play);

        match = new Match(matchGames, x, o);
        play = match.play();
        System.out.println("Random " + play);

    }
}
