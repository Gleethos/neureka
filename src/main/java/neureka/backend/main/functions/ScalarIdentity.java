package neureka.backend.main.functions;

public final class ScalarIdentity implements ScalarFun
{
    @Override public String id() { return "idy"; }

    @Override public String activationCode() { return "output = input; \n"; }

    @Override public String derivationCode() { return "output = 1.0f; \n"; }

    @Override
    public CPUFun getActivation() {
        return new CPUFun() {
            @Override public double invoke(double x)  { return x; }
            @Override public float invoke(float x)   { return x; }
            @Override public int invoke(int x)     { return x; }
            @Override public long invoke(long x)    { return x; }
            @Override public boolean invoke(boolean x) { return x; }
            @Override public char invoke(char x)    { return x; }
            @Override public Object invoke(Object x)  { return x; }
        };
    }

    @Override
    public CPUFun getDerivative() {
        return new CPUFun() {
            @Override public double invoke(double x) { return 1; }
            @Override public float invoke(float x)  { return 1; }
            @Override public int invoke(int x)    { return 1; }
            @Override public long invoke(long x)   { return 1; }
            @Override public Object invoke(Object x) { return null; }
        };
    }

}
