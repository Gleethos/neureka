package neureka.utility;

import java.util.ArrayList;
import java.util.Random;

public class DataHelper
{
    public static double getDoubleOf(long seed) {
        double newNumber;
        double divisor = 0;
        divisor = (Math.pow(10, getIntegerOf(seed - 1, 2)) / Math.pow(10, getIntegerOf(seed + 1, 2)));
        final int[] prims = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61};
        int rank = (int) Math.abs(seed % prims.length);
        int min = 3;
        newNumber = 0;
        while (newNumber == 0 || min >= 0) {
            newNumber /= 2;
            newNumber += (Math.abs((getIntegerOf(prims[rank] + seed - 4, 10))));
            if (divisor < 10) {
                divisor *= 10;
            } else {
                divisor /= 10;
            }
            newNumber /= divisor;
            if (newNumber != 0) {
                min--;
            }
            rank++;
            rank = Math.abs(rank % prims.length);
        }
        newNumber = Math.abs(newNumber);
        newNumber *= (Math.abs(getIntegerOf(seed + 4 - prims[rank], 100_000))<50_106)?-1:1;
        return newNumber;
    }

    public static int getIntegerOf(long seed, int m) {
        int[] prims = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61};
        long rank = Double.doubleToRawLongBits(seed * Math.pow(61*Math.abs(seed % prims.length), 0.5));
        return getIntegerOf(seed, m, (int)(rank ^ (rank >>> 32)));
    }

    public static int getIntegerOf(long seed, int m, int rank) {
        int[] prims = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61};
        rank = Math.abs(rank % prims.length);
        seed = (seed == 0)?6731299924898493759L:seed;
        seed *= (((prims[rank] * seed + m) % 2 == 0)?-1:1);
        seed =  (4438274645895732L * seed * Double.doubleToRawLongBits(Math.pow(prims[rank], 0.5)) + seed);
        return ((int) (seed ^ (seed >>> 32)))%m;
    }

    public static double randomDouble() {
        Random dice = new Random();
        double newNumber;
        double divisor = 0;
        divisor = (Math.pow(10, (dice.nextInt() % 2)) / Math.pow(10, (dice.nextInt() % 2)));
        int min = 3;
        newNumber = 0;
        while (newNumber == 0 || min >= 0) {
            newNumber /= 2;
            newNumber += ((dice.nextInt(21) - 10)) * (dice.nextInt(3) - 1);
            if (divisor < 10) {
                divisor *= 10;
            } else {
                divisor /= 10;
            }
            newNumber /= divisor;
            if (newNumber != 0) {
                min--;
            }
        }
        return newNumber;
    }

    public static float[] doubleToFloat(double[] data){
        if(data==null){
            return null;
        }
        float[] newData = new float[data.length];
        for(int i=0; i<data.length; i++) newData[i] = (float) data[i];
        return newData;
    }

    public static double[] floatToDouble(float[] data){
        if(data==null){
            return null;
        }
        double[] newData = new double[data.length];
        for(int i=0; i<data.length; i++) newData[i] = (double)data[i];
        return newData;
    }


}
