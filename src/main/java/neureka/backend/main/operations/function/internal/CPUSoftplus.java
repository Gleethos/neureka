package neureka.backend.main.operations.function.internal;

/**
 *  SoftPlus is a smooth approximation to the ReLU function and can be used
 *  to constrain the output of a machine to always be positive.
 */
public final class CPUSoftplus implements ActivationFun
{
    @Override public String id() { return "softplus"; }

    @Override public String activationCode() { return "output = log( 1.0f + exp( input ) );"; }

    @Override public String derivationCode() { return "output = 1.0f / ( 1.0f + exp( -input ) );\n"; }

    @Override public double activate(double x) { return Math.log( 1d + Math.exp( x ) ); }

    @Override public double derive(double x) { return 1d / ( 1d + Math.exp( -x ) ); }

}
