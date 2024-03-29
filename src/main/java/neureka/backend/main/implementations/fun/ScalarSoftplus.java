package neureka.backend.main.implementations.fun;

import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;

/**
 *  SoftPlus is a smooth approximation to the ReLU function and can be used
 *  to constrain the output of a machine to always be positive.
 */
public final class ScalarSoftplus implements ScalarFun
{
    @Override public String id() { return "softplus"; }

    @Override public String activationCode() { return "output = log( 1.0f + exp( input ) );"; }

    @Override public String derivationCode() { return "output = 1.0f / ( 1.0f + exp( -input ) );\n"; }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double invoke(double x) { return Math.log( 1d + Math.exp( x ) ); }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override public double invoke(double x) { return 1d / ( 1d + Math.exp( -x ) ); }
        };
    }

}
