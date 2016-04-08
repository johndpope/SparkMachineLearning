package com.lordjoe.machine_learning.tictactoe;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * com.lordjoe.machine_learning.tictactoe.Board
 * User: Steve
 * Date: 4/6/2016
 */
public class Board implements Comparable<Board> {
    private static  int gBoardSize = 3;

    public static void setgBoardSize(int size) {
        gBoardSize = size;
    }

    public static int getBoardSize() {
        return gBoardSize;
    }

    public static  int getTotalBoardSize()
    {
        return getBoardSize() * getBoardSize();
    }


    private Player[][] moves = new Player[getBoardSize()][getBoardSize()];
    private transient String stringRepresentation;
    private   transient List<Row> rows;   // build just once np need tp serialize

    public Board() {
    }

    public Board(String s) {
        int index = 0;
        for (int i = 0; i < getBoardSize(); i++) {
            for (int j = 0; j < getBoardSize(); j++) {
                String player = s.substring(index, index + 1);
                if (!(" ".equals(player))) {
                    moves[i][j] = Player.valueOf(player);
                }
                index++;
            }
        }
    }

    public Board(Board s) {
        int index = 0;
        for (int i = 0; i < getBoardSize(); i++) {
            for (int j = 0; j < getBoardSize(); j++) {
                moves[i][j] = s.moves[i][j];
            }
        }
    }

    /**
     * could this be produced by adding more moves to s
     *
     * @param s
     * @return
     */
    public boolean isSubBoard(Board s) {
        for (int i = 0; i < getBoardSize(); i++) {
            for (int j = 0; j < getBoardSize(); j++) {
                if (moves[i][j] != null)
                    if (moves[i][j] != s.moves[i][j])
                        return false; // different
            }
        }
        return true;
    }

    public Player getCell(int x, int y) {
        return moves[x][y];
    }

    public Player getCell(Position p) {
        return getCell(p.x, p.y);
    }


    public boolean isFull() {
        return getLegalMoves().size() == 0;
    }

    /**
     * find all possible moves
     *
     * @return
     */
    public List<Position> getLegalMoves() {
        List<Position> holder = new ArrayList<Position>();
        for (int i = 0; i < getBoardSize(); i++) {
            for (int j = 0; j < getBoardSize(); j++) {
                if (moves[i][j] == null)
                    holder.add(new Position(i, j));
            }
        }
        return holder;
    }

    /**
     * put a player token at a position
     * @param p non null player
     * @param pos  non null not occupied position
     */
    public void makeMove(Player p, Position pos) {
        if (moves[pos.x][pos.y] != null)
            throw new IllegalArgumentException("Illegal move - position occupied");
        moves[pos.x][pos.y] = p;
        stringRepresentation = null;
    }

    /**
     * find the first winning position and return it or null if no winning
     * position
        * @return as above
     */
    public Player hasWin() {
        List<Row> rows = getRows();
        for (Row row : rows) {
           Player winner = row.getWinner();
            if(winner != null)
                return winner;
        }
        return null;
    }

    /**
     * return a string representing draw, the winning player or ""
     * @return
     */
    public String getWinningString() {
        Player p = hasWin();
        if (p != null)
            return p.toString();
        if (isFull())
            return "draw";
        throw new IllegalStateException("not determined");
    }

    /**
     * get horizontal, vertical and diagonal rows of this board
     * cache the answer
     * @return
     */
    public List<Row> getRows()
    {
        if(rows == null)  {
            List<Row> holder = new ArrayList<Row>();
            holder.addAll(getDiagonals()) ;
            holder.addAll(getHorizontals()) ;
            holder.addAll(getVerticals()) ;
            rows = holder;
        }
        return rows;
    }

    /**
     * intermediate function
     * @return
     */
    protected List<Row> getDiagonals() {
        List<Row> holder = new ArrayList<Row>();
        List<Position> up = new ArrayList<>();
        List<Position> down = new ArrayList<>();
        for (int i = 0; i < getBoardSize(); i++) {
            up.add(new Position(i, i));
            down.add(new Position(i, getBoardSize() - 1 - i));
        }
        holder.add(new Row(this, up));
        holder.add(new Row(this, down));
        return holder;
    }


    protected List<Row> getHorizontals() {
        List<Row> holder = new ArrayList<Row>();
        List<Position> positionHolder = new ArrayList<>();
        for (int i = 0; i < getBoardSize(); i++) {
            for (int j = 0; j < getBoardSize(); j++) {
                positionHolder.add(new Position(i, j));
            }
            holder.add(new Row(this, positionHolder));
            positionHolder.clear();

        }
        return holder;
    }

    protected List<Row> getVerticals() {
        List<Row> holder = new ArrayList<Row>();
        List<Position> positionHolder = new ArrayList<>();
        for (int i = 0; i < getBoardSize(); i++) {
            for (int j = 0; j < getBoardSize(); j++) {
                positionHolder.add(new Position(j, i));
            }
            holder.add(new Row(this, positionHolder));
            positionHolder.clear();
         }
        return holder;
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        return toString().equals(o.toString());
    }

    @Override
    public int hashCode() {
        return toString().hashCode();
    }

    @Override
    public String toString() {
        if (stringRepresentation == null) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < getBoardSize(); i++) {
                for (int j = 0; j < getBoardSize(); j++) {
                    if (moves[i][j] == null)
                        sb.append(" ");
                    else
                        sb.append(moves[i][j].toString());
                }
            }
            stringRepresentation = sb.toString();
        }
        return stringRepresentation;
    }

    public String toPrettyPrintString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < getBoardSize(); i++) {
            for (int j = 0; j < getBoardSize(); j++) {
                if (moves[i][j] == null)
                    sb.append(" ");
                else
                    sb.append(moves[i][j].toString());
            }
            sb.append("\n");
        }


        return sb.toString();
    }

    /**
     * compare by string representation
     *
     * @param o
     * @return
     */
    @Override
    public int compareTo(Board o) {
        return toString().compareTo(o.toString());
    }
}
