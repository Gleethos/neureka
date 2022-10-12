package neureka.backend.main.implementations.fun;

import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;

public class ScalarExp implements ScalarFun
{
    @Override public String id() { return "exp"; }

    @Override
    public String activationCode() {
        return "output = exp( input );\n";
    }

    @Override
    public String derivationCode() {
        return "output = exp( input );\n";
    }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double invoke(double x) { return Math.exp( x ); }
            @Override public float invoke(float x) { return (float) Math.exp( x ); }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override public double invoke(double x) { return Math.exp( x ); }
            @Override public float invoke(float x) { return (float) Math.exp( x ); }
        };
    }

}
