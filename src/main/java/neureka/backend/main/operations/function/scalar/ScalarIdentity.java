package neureka.backend.main.operations.function.scalar;

public final class ScalarIdentity implements ScalarFun
{
    @Override public String id() { return "idy"; }

    @Override public String activationCode() { return "output = input; \n"; }

    @Override public String derivationCode() { return "output = 1.0f; \n"; }

    @Override public double activate(double x) { return x; }

    @Override public double derive(double x) { return 1; }

    @Override public float activate(float x) { return x; }

    @Override public float derive(float x) { return 1; }

    @Override public int activate(int x) { return x; }

    @Override public int derive(int x) { return 1; }

    @Override public long activate(long x) { return x; }

    @Override public long derive(long x) { return 1; }

    @Override public boolean activate(boolean x) { return x; }

    @Override public char activate(char x) { return x; }

    public Object activate(Object x) { return x; }

    public Object derive(Object x) { return null; }

}
