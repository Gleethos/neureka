package neureka.backend.main.functions;

public interface CPUFun {

    double invoke(double x);

    default float invoke(float x) { return (float) invoke( (double) x ); }

    default int invoke(int x) { return (int) Math.round( invoke( (double) x ) ); }

    default long invoke(long x) { return Math.round( invoke( (double) x ) ); }

    default byte invoke(byte x) { return (byte) Math.round( invoke( (double) x ) ); }

    default short invoke(short x) { return (short) Math.round( invoke( (double) x ) ); }

    default boolean invoke(boolean x) { return Math.round( invoke( x ? 1 : 0 ) ) != 0; } // Some default behaviors, it might make sense to override this for some activations.

    default char invoke(char x) { return (char) Math.round( invoke( (int) x ) ); } // Some default behaviors, it might make sense to override this for some activations.

    default Object invoke(Object x) { throw new IllegalStateException("Not implemented for operation "+getClass().getSimpleName()); }

}
