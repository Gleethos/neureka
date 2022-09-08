package neureka.backend.main.functions;

public final class ScalarReLU implements ScalarFun
{
    @Override public String id() { return "relu"; }

    @Override
    public String activationCode() {
        return "if (input >= 0) {  output = input; } else { output = input * (float)0.01; }\n";
    }

    @Override
    public String derivationCode() {
        return "if (input >= 0) { output = (float)1; } else { output = (float)0.01; }\n";
    }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double invoke(double x) { return ( x >= 0 ? x : x * .01 ); }
            @Override public float invoke(float x) { return ( x >= 0 ? x : x * .01f ); }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override public double invoke(double x) { return ( x >= 0 ? 1 : .01); }
            @Override public float invoke(float x) { return ( x >= 0 ? 1f : .01f ); }
        };
    }

}
