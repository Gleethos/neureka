package neureka.optimization.implementations;

import neureka.Tsr;
import neureka.optimization.OptimizerFactory;

public class SGDFactory implements OptimizerFactory
{
    private final double _learningRate;

    public SGDFactory() { _learningRate = 0.01; }

    // The copy constructor should be private, use withers instead!
    private SGDFactory(double learningRate) {
        _learningRate = learningRate;
    }

    // Withers:

    public SGDFactory withLearningRate(double learningRate) {
        return new SGDFactory(learningRate);
    }

    @Override
    public <V extends Number> SGD<V> create(Tsr<V> target) {
        return new SGD<>(_learningRate);
    }

}
