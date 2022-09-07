package neureka.backend.main.operations.function.scalar;

public final class ScalarAbsolute implements ScalarFun
{
    @Override public String id() { return "abs"; }

    @Override public String activationCode() { return "output = fabs( input );\n"; }

    @Override public String derivationCode() { return "output = ( input < 0 ) ? -1 : 1;\n"; }

    @Override public double activate(double x) { return Math.abs( x ); }

    @Override public double derive(double x) { return ( x < 0d ? -1d : 1d ); }

    @Override public float activate(float x) { return Math.abs( x ); }

    @Override public float derive(float x) { return ( x < 0f ? -1f : 1f ); }

    @Override public int activate(int x) { return Math.abs( x ); }

    @Override public int derive(int x) { return ( x < 0 ? -1 : 1 ); }
}
