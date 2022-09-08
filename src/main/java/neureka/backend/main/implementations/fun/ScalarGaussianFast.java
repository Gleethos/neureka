package neureka.backend.main.implementations.fun;

import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;

public class ScalarGaussianFast implements ScalarFun
{
    @Override public String id() { return "fast_gaus"; }

    @Override public String activationCode() { return "output = 1 / ( 1 + input * input );\n"; }

    @Override public String derivationCode() {
        return "float x2 = input * input;\n" +
               "output = -2 * input / ( x2 * x2 + 2 * x2 + 1 );\n";
    }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double invoke(double x) { return 1 / ( 1 + x * x ); }
            @Override public float invoke(float x) { return 1 / ( 1 + x * x ); }

        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override public double invoke(double x) {
                double x2 = x * x;
                return  -2 * x / ( x2 * x2 + 2 * x2 + 1 );
            }
            @Override public float invoke(float x) {
                float x2 = x * x;
                return  -2 * x / ( x2 * x2 + 2 * x2 + 1 );
            }
        };
    }

}
