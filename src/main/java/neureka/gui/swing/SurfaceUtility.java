package neureka.gui.swing;


public class SurfaceUtility {

    public double[] getCurvePointOn(double t, double[] point) {
        if (point.length % 2 == 1) {
            return null;
        }
        double[] result = {0, 0};
        int n = point.length / 2 - 1;
        for (int i = 0; i <= n; i++) {
            result[0] += point[(2 * i)] * choose(n, i) * Math.pow((1 - t), i) * Math.pow((t), n - i);
            result[1] += point[(2 * i) + 1] * choose(n, i) * Math.pow((1 - t), i) * Math.pow((t), n - i);
        }
        return result;
    }

    public int choose(int n, int k) {
        int nCk = 1;
        for (int ki = 0; ki < k; ki++) {
            nCk = nCk * (n - ki) / (ki + 1);
        }
        return nCk;
    }

    public static double magnitudeOf(double vecX, double vecY) {
        return Math.pow(Math.pow(vecX, 2) + Math.pow(vecY, 2), 0.5);
    }

    public double unitaryVectorProduct(double firstX, double firstY, double secX, double secY) {
        double d = magnitudeOf(firstX, firstY) * magnitudeOf(secX, secY);
        return (firstX * secX + firstY * secY) / d;
    }


}
