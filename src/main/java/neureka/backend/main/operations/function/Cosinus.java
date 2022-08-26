package neureka.backend.main.operations.function;

public final class Cosinus extends AbstractActivationOperation
{
    public Cosinus() { super( "cos" ); }

    @Override
    protected String _activationCode() { return "output = cos( input );\n"; }

    @Override
    protected String _derivationCode() { return "output = -sin( input );\n"; }

    @Override
    protected double _activate(double x) { return Math.cos(x); }

    @Override
    protected double _derive(double x) { return -Math.sin(x); }

}
