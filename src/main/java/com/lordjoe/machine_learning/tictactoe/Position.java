package com.lordjoe.machine_learning.tictactoe;

/**
 * com.lordjoe.machine_learning.tictactoe.Position
 * a single x,y value
 * User: Steve
 * Date: 4/6/2016
 */
public class Position implements  Comparable<Position>{
    public final int x;
    public final int y;

    public Position(int x, int y) {
        if(x < 0 || x >= TicTacToeBoard.getBoardSize())
            throw new IllegalArgumentException("bad x");
        if(y < 0 || y >= TicTacToeBoard.getBoardSize())
            throw new IllegalArgumentException("bad y");
        this.x = x;
        this.y = y;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Position position = (Position) o;

        if (x != position.x) return false;
        return y == position.y;

    }

    @Override
    public int hashCode() {
        int result = x;
        result = 31 * result + y;
        return result;
    }

    /**
     * sort by x then x
     * @param o
     * @return
     */
    @Override
    public int compareTo(Position o) {
        if(x != o.x)
            return Integer.compare(x ,o.x);
        if(y != o.y)
            return Integer.compare(y ,o.y);
         return 0;
    }

    @Override
    public String toString() {
        return "" + x + "," + y ;
    }
}
