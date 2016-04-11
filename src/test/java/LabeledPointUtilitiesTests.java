import org.apache.spark.mllib.regression.LabeledPoint;

import java.io.*;
import java.util.List;

/**
 * PACKAGE_NAME.LabeledPointUtilitiesTests
 * User: Steve
 * Date: 3/15/2016
 */
public class LabeledPointUtilitiesTests {

    public static void main(String[] args) {
        File inFile = new File(args[0]);
        File outFile = new File(args[1]);
        double[] keey = new double[args.length - 2];
        for (int i = 0; i < keey.length; i++) {
            keey[i] = Double.parseDouble(args[i + 2]);

        }
        saveLabels(inFile, outFile, keey);
    }


    public void saveLabels(List<LabeledPoint> labels, File outFile) {
        PrintWriter out = null;
        try {
            out = new PrintWriter((new FileWriter(outFile)));
            for (LabeledPoint p : labels) {
                out.println(p.toString());
            }
        } catch (IOException e) {
            throw new RuntimeException(e);

        } finally {
            if (out != null)
                out.close();
        }
     }

    private static void saveLabels(File inFile, File outFile, double[] keey) {
        LineNumberReader inp = null;
        PrintWriter out = null;
        try {
            inp = new LineNumberReader((new FileReader(inFile)));
            out = new PrintWriter((new FileWriter(outFile)));
            String line = inp.readLine();
            while (line != null) {
                LabeledPoint parse = LabeledPoint.parse(line);
                for (int i = 0; i < keey.length; i++) {
                    double v = keey[i];
                    if (Math.abs(v - parse.label()) < 0.001) {
                        out.println(line);
                        break;
                    }
                }
                line = inp.readLine();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);

        } finally {
            if (inp != null)
                try {
                    inp.close();
                } catch (IOException e) {
                }
            if (out != null)
                out.close();
        }

    }

}
