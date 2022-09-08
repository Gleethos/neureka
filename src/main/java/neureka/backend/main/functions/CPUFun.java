package neureka.backend.main.functions;

public interface CPUFun {

    double activate(double x);

    default float activate(float x) { return (float) activate( (double) x ); }

    default int activate(int x) { return (int) Math.round( activate( (double) x ) ); }

    default long activate(long x) { return Math.round( activate( (double) x ) ); }

    default byte activate(byte x) { return (byte) Math.round( activate( (double) x ) ); }

    default short activate(short x) { return (short) Math.round( activate( (double) x ) ); }

    default boolean activate(boolean x) { return Math.round( activate( x ? 1 : 0 ) ) != 0; } // Some default behaviors, it might make sense to override this for some activations.

    default char activate(char x) { return (char) Math.round( activate( (int) x ) ); } // Some default behaviors, it might make sense to override this for some activations.

    default Object activate(Object x) { throw new IllegalStateException("Not implemented for operation "+getClass().getSimpleName()); }

}
