package com.lordjoe.machine_learning.tictactoe;

import java.util.ArrayList;
import java.util.List;

/**
 * com.lordjoe.machine_learning.tictactoe.Match
 * User: Steve
 * Date: 4/7/2016
 */
public class Match {

    private final List<TicTatToeGame> games = new ArrayList<>();
    private final IStrategy x;
    private final IStrategy o;
    private final boolean showProgress;
    private final double number;
    private int draws = 0;
    private int xwins = 0;
    private int owins = 0;


    public Match(int ngames, IStrategy x, IStrategy o, boolean showProgress) {
        this.o = o;
        this.x = x;
        this.showProgress = showProgress;
        for (int i = 0; i < ngames; i++) {
            games.add(new TicTatToeGame(x, o));

        }
        number = ngames;
    }
    public Match(int ngames, IStrategy x, IStrategy o ) {
         this(ngames,x,o,false) ;
    }

    public String play() {
        int numberPlayed = 0;
        for (TicTatToeGame game : games) {
            Player player = game.playGame();
            if (player == null)
                draws++;
            else {
                switch (player) {
                    case O:
                        owins++;
                        break;
                    case X:
                        xwins++;
                        break;
                }
             }
            numberPlayed++;
            if(showProgress && numberPlayed %  10  == 0)
                System.out.println("played " + numberPlayed);
        }
        return "x " + xwins / number + " o " + owins / number + " draw " + draws / number;
    }

    public int getDraws() {
        return draws;
    }

    public int getXwins() {
        return xwins;
    }

    public int getOwins() {
        return owins;
    }
}
