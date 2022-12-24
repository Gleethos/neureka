package neureka.optimization.implementations;

import neureka.Tsr;
import neureka.optimization.OptimizerFactory;

public class AdaGradFactory implements OptimizerFactory 
{
    private final double _learningRate;

    public AdaGradFactory() { _learningRate = 0.01; }

    // The copy constructor should be private, use withers instead!
    private AdaGradFactory(double learningRate) {
        _learningRate = learningRate;
    }

    // Withers:

    public AdaGradFactory withLearningRate(double learningRate) {
        return new AdaGradFactory(learningRate);
    }

    @Override
    public <V extends Number> AdaGrad<V> create(Tsr<V> target) {
        return new AdaGrad<>(target, _learningRate);
    }

}
