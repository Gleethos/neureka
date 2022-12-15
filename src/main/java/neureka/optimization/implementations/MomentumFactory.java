package neureka.optimization.implementations;

import neureka.Tsr;
import neureka.optimization.OptimizerFactory;

public class MomentumFactory implements OptimizerFactory 
{
    private final double _learningRate;
    private final double _decayRate;

    public MomentumFactory() {
        _learningRate = 0.01;
        _decayRate = 0.9;
    }

    // The copy constructor should be private, use withers instead!
    private MomentumFactory(double learningRate, double decayRate) {
        _learningRate = learningRate;
        _decayRate = decayRate;
    }

    // Withers:

    public MomentumFactory withLearningRate(double learningRate) {
        return new MomentumFactory(learningRate, _decayRate);
    }
    
    public MomentumFactory withDecayRate(double decayRate) {
        return new MomentumFactory(_learningRate, decayRate);
    }

    @Override
    public <V extends Number> Momentum<V> create(Tsr<V> target) {
        return new Momentum<>(target, _learningRate, _decayRate);
    }
    
}
