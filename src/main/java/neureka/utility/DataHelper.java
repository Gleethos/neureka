package neureka.utility;

import java.util.Random;

public class DataHelper
{

    public static double[] newSeededDoubleArray(String seed, int size){
        return newSeededDoubleArray(_longStringHash(seed), size);
    }

    public static double[] newSeededDoubleArray(long seed, int size){
        return seededDoubleArray(new double[size], seed);
    }

    public static double[] seededDoubleArray(double[] array, String seed){
        return seededDoubleArray(array, _longStringHash(seed));
    }

    public static double[] seededDoubleArray(double[] array, long seed){
        Random dice = new Random();
        dice.setSeed(seed);
        for(int i=0; i<array.length; i++) array[i] = dice.nextGaussian();
        return array;
    }

    public static float[] newSeededFloatArray(String seed, int size){
        return newSeededFloatArray(_longStringHash(seed), size);
    }

    public static float[] newSeededFloatArray(long seed, int size){
        return seededFloatArray(new float[size], seed);
    }

    public static float[] seededFloatArray(float[] array, String seed){
        return seededFloatArray(array, _longStringHash(seed));
    }

    public static float[] seededFloatArray(float[] array, long seed){
        Random dice = new Random();
        dice.setSeed(seed);
        for(int i=0; i<array.length; i++) array[i] = (float)dice.nextGaussian();
        return array;
    }

    private static long _longStringHash(String string)
    {
        long h = 1125899906842597L; // prime
        int len = string.length();
        for (int i = 0; i < len; i++) h = 31*h + string.charAt(i);
        return h;
    }

    public static float[] doubleToFloat(double[] data){
        if(data==null) return null;
        float[] newData = new float[data.length];
        for(int i=0; i<data.length; i++) newData[i] = (float) data[i];
        return newData;
    }

    public static double[] floatToDouble(float[] data){
        if(data==null) return null;
        double[] newData = new double[data.length];
        for(int i=0; i<data.length; i++) newData[i] = (double)data[i];
        return newData;
    }


}
