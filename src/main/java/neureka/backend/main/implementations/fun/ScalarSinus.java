package neureka.backend.main.implementations.fun;

import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;

public final class ScalarSinus implements ScalarFun
{
    @Override public String id() { return "sin"; }

    @Override public String activationCode() { return "output = sin( input );\n"; }

    @Override public String derivationCode() { return "output = cos( input );\n"; }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double invoke(double x) { return Math.sin(x); }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override public double invoke(double x) { return Math.cos(x); }
        };
    }

}
