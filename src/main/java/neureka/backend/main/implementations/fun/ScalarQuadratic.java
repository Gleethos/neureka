package neureka.backend.main.implementations.fun;

import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;

public final class ScalarQuadratic implements ScalarFun
{
    @Override public String id() { return "quad"; }

    @Override public String activationCode() { return "output = input * input;\n"; }

    @Override public String derivationCode() { return "output = 2 * input;\n"; }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double invoke(double x) { return x * x; }
            @Override public float invoke(float x) { return x * x; }
            @Override public int invoke(int x) { return x * x; }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override public double invoke(double x) { return 2 * x; }
            @Override public float invoke(float x) { return 2 * x; }
            @Override public int invoke(int x) { return 2 * x; }
        };
    }

}
