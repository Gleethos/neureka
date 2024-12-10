package neureka.backend.main.implementations.fun.api;

public strictfp interface CPUBiFun
{
    double invoke(double a, double b);

    default float invoke(float a, float b) {
        return (float) invoke( a, (double) b );
    }

    default int invoke(int a, int b) {
        return (int) Math.round( invoke( a, (double) b ) );
    }

    default long invoke(long a, long b) {
        return Math.round( invoke( (double) a, (double) b ) );
    }

    default byte invoke(byte a, byte b) {
        return (byte) Math.round( invoke( a, (double) b ) );
    }

    default short invoke(short a, short b) {
        return (short) Math.round( invoke( a, (double) b ) );
    }

    default boolean invoke(boolean a, boolean b) {
        return invoke( a ? 1 : 0, b ? 1 : 0 ) != 0; // Some default behaviors, it might make sense to override this for some activations.
    }

    default char invoke(char a, char b) {
        return (char) invoke( a, (int) b ); // Some default behaviors, it might make sense to override this for some activations.
    }

    default Object invoke(Object a, Object b) {
        throw new IllegalStateException("Not implemented for operation "+getClass().getSimpleName());
    }
}
