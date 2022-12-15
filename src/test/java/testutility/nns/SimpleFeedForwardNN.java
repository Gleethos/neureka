package testutility.nns;

import neureka.Neureka;
import neureka.Tsr;
import neureka.math.Function;
import neureka.optimization.Optimizer;
import neureka.optimization.implementations.ADAM;

public class SimpleFeedForwardNN
{
    private final Tsr<Float> _W1, _W2, _W3, _b1, _b2, _b3;
    private final Function _a1 = Neureka.get().backend().getAutogradFunction().gelu();
    private final Function _a2 = Neureka.get().backend().getAutogradFunction().gelu();
    private final Function _a3 = Neureka.get().backend().getAutogradFunction().tanh();

    private Tsr<Float> _lastPrediction = null;

    public SimpleFeedForwardNN(int size, int seed ) {
        int size1 = (int) ( size * 1.2 );
        int size2 = (int) ( size * 1.1 );
        _W1 = Tsr.ofFloats().withShape(size,  size1).andSeed(++seed).setRqsGradient(true);
        _W2 = Tsr.ofFloats().withShape(size1, size2).andSeed(++seed).setRqsGradient(true);
        _W3 = Tsr.ofFloats().withShape(size2, size ).andSeed(++seed).setRqsGradient(true);
        _b1 = Tsr.ofFloats().withShape(1,     size1).andSeed(++seed).setRqsGradient(true);
        _b2 = Tsr.ofFloats().withShape(1,     size2).andSeed(++seed).setRqsGradient(true);
        _b3 = Tsr.ofFloats().withShape(1,     size ).andSeed(++seed).setRqsGradient(true);
        _W1.set(Optimizer.ADAM);
        _W2.set(Optimizer.ADAM);
        _W3.set(Optimizer.ADAM);
        _b1.set(Optimizer.ADAM);
        _b2.set(Optimizer.ADAM);
        _b3.set(Optimizer.ADAM);
    }

    public Tsr<Float> forward( Tsr<Float> x ) {
        Tsr<Float> z1 = _a1.call( x.matMul(_W1).plus(_b1) );
        Tsr<Float> z2 = _a2.call( z1.matMul(_W2).plus(_b2) );
        Tsr<Float> z3 = _a3.call( z2.matMul(_W3).plus(_b3) );
        _lastPrediction = z3;
        return z3;
    }

    public double train( Tsr<Float> y ) {
        if ( _lastPrediction == null )
            throw new IllegalStateException("No prediction made yet!");

        Tsr<Float> error = y.minus(_lastPrediction).power(2f).sum();
        error.backward();
        _lastPrediction = null;
        _applyGradients();
        return error.item();
    }

    private void _applyGradients() {
        _W1.applyGradient();
        _W2.applyGradient();
        _W3.applyGradient();
        _b1.applyGradient();
        _b2.applyGradient();
        _b3.applyGradient();
    }

}
