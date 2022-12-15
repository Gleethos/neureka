package neureka.optimization.implementations;

import neureka.Tsr;
import neureka.optimization.OptimizerFactory;

public class RMSPropFactory implements OptimizerFactory
{
    private final double _learningRate;
    private final double _decayRate;

    public RMSPropFactory() {
        _learningRate = 0.001;
        _decayRate = 0.9;
    }

    // The copy constructor should be private, use withers instead!
    private RMSPropFactory(double learningRate, double decayRate) {
        _learningRate = learningRate;
        _decayRate = decayRate;
    }

    // Withers:

    public RMSPropFactory withLearningRate(double learningRate) {
        return new RMSPropFactory(learningRate, _decayRate);
    }

    public RMSPropFactory withDecayRate(double decayRate) {
        return new RMSPropFactory(_learningRate, decayRate);
    }

    @Override
    public <V extends Number> RMSProp<V> create(Tsr<V> target) {
        return new RMSProp<>((Tsr<Number>) target, _learningRate, _decayRate);
    }

}
