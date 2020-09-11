package neureka.optimization.implementations;

import neureka.Tsr;
import neureka.calculus.Function;
import neureka.optimization.Optimizer;

public class SGD implements Optimizer
{
    private final double _learningRate;
    private final Function _function;

    public SGD( double leaningRate )
    {
        _learningRate = leaningRate;
        _function = Function.create("I[0] <- (-1 * (I[0] - "+leaningRate+"))", false);
    }

    @Override
    public void optimize( Tsr w ) {
        Tsr g = w.find(Tsr.class);
        Function.Detached.IDY.call(_function.call( g ));
    }

    public double learningRate(){
        return _learningRate;
    }

    @Override
    public void update( Tsr oldOwner, Tsr newOwner ) {

    }
}
