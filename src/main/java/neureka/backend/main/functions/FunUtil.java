package neureka.backend.main.functions;

class FunUtil
{
    /**
     *  This is extremely fast and has virtually the same accuracy as {@code 1 / Math.pow(x, 0.5)}.
     */
    public static double invSqrt(double x) {
        double xhalf = 0.5d * x;
        long i = Double.doubleToLongBits(x);
        i = 0x5fe6ec85e7de30daL - (i >> 1);
        x = Double.longBitsToDouble(i);
        x *= (1.5d - xhalf * x * x);
        x *= (1.5d - xhalf * x * x); // more accuracy...
        x *= (1.5d - xhalf * x * x); // more accuracy...
        x *= (1.5d - xhalf * x * x); // more accuracy...
        return x;
    }

    public static float invSqrt( float x ) {
        float xhalf = 0.5f * x;
        int i = Float.floatToIntBits(x);
        i = 0x5f3759df - (i >> 1);
        x = Float.intBitsToFloat(i);
        x *= (1.5f - xhalf * x * x);
        x *= (1.5f - xhalf * x * x);
        return x;
    }

}
