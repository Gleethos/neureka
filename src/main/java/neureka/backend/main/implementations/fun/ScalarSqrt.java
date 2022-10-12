package neureka.backend.main.implementations.fun;

import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;

public class ScalarSqrt implements ScalarFun
{
    @Override public String id() { return "sqrt"; }

    @Override
    public String activationCode() {
        return "output = sqrt( input );\n";
    }

    @Override
    public String derivationCode() {
        return "output = 0.5f * fast_inverse_sqrt( input );\n";
    }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double invoke( double x ) { return Math.sqrt( x ); }
            @Override public float invoke( float x ) { return (float) Math.sqrt( x ); }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override public double invoke(double x) { return 0.5 * FunUtil.invSqrt( x ); }
            @Override public float invoke(float x) { return 0.5f * FunUtil.invSqrt( x ); }
        };
    }

}
