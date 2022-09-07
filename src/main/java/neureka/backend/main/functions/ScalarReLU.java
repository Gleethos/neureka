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
    public double activate(double x) { return ( x >= 0 ? x : x * .01 ); }

    @Override
    public float activate(float x) { return ( x >= 0 ? x : x * .01f ); }

    @Override
    public double derive(double x) { return ( x >= 0 ? 1 : .01); }

    @Override
    public float derive(float x) { return ( x >= 0 ? 1f : .01f ); }

}
