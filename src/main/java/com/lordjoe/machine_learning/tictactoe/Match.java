package com.lordjoe.machine_learning.tictactoe;

import java.util.ArrayList;
import java.util.List;

/**
 * com.lordjoe.machine_learning.tictactoe.Match
 * User: Steve
 * Date: 4/7/2016
 */
public class Match {

    private final List<Game> games = new ArrayList<>();
    private final IPlayer x;
    private final IPlayer o;
    private final double number;
    private int draws = 0;
    private int xwins = 0;
    private int owins = 0;

    public Match(int ngames, IPlayer x, IPlayer o) {
        this.o = o;
        this.x = x;
        for (int i = 0; i < ngames; i++) {
            games.add(new Game(x, o));

        }
        number = ngames;
    }

    public String play() {

        for (Game game : games) {
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
        }
        return "x " + xwins / number + " o " + owins / number + " draw " + draws / number;
    }
}
