package com.lordjoe.machine_learning.tictactoe;



import java.util.*;

/**
 * com.lordjoe.machine_learning.tictactoe.Game
 * a game between two players - X always moves first (X and O are arbitrary and
 *  really mean forst and second player
 * User: Steve
 * Date: 4/6/2016
 */
public class Game {

    private  final Board board = new Board();
    private  final Map<Player,IPlayer>  players = new HashMap<Player,IPlayer>();
    private Player mover = Player.X; // X always starts
    private Player winner;


    public Game(IPlayer x,IPlayer y) {
        if(x.getPlayer() == y.getPlayer())   throw new IllegalArgumentException("players must be different");
        players.put(x.getPlayer(),x);
        players.put(y.getPlayer(),y);
    }



    /**
     * true if there is a winner or the board is full
     * @return
     */
    public boolean isFinished()
    {
        if(winner != null)
            return true;
        return board.isFull();
    }

    /**
     * return the winner - first found or null if no winner
     * makeMove will set winner
     * @return
     */
    public Player getWinner() {
        return winner;
    }

    /**
     * make a single move - this resets mover and
     * may set winner
     * @return  true if there is a winner or the board is full
     */
    public boolean makeMove()
    {
        if(winner != null)
            throw new IllegalStateException("no moves allowed if game has a winner");

        Position position = players.get(mover).choseMove(board);
        board.makeMove(mover,position);
        winner = board.hasWin();
        if(isFinished())  {
            mover = null;
            return true; // game over;
        }
        mover = mover.next();
         return false; // still on
    }

    /**
     * X plays then Y until finished
     * @return
     */
    public Player playGame()
    {
        Player mover = Player.X;
        while(!isFinished())  {
            makeMove();
        }
        return winner;
     }

    /**
     * record all moves from first until winner or draw
     * @return
     */
    public List<Board>  getPlayBoards()
    {
        List<Board> holder = new ArrayList<Board>();

        Player mover = Player.X;
        while(!isFinished())  {
            makeMove();
            holder.add(getBoard());
        }

        return holder;
    }



    public Board getBoard() {
        return new Board(board);
      }
}
