package neureka.optimization.implementations;

import neureka.Tsr;
import neureka.optimization.OptimizerFactory;

public class ADAMFactory implements OptimizerFactory 
{
    private final double _learningRate;

    public ADAMFactory() { _learningRate = 0.01; }

    // The copy constructor should be private, use withers instead!
    private ADAMFactory(double learningRate) {
        _learningRate = learningRate;
    }
    
    // Withers:

    public ADAMFactory withLearningRate(double learningRate) {
        return new ADAMFactory(learningRate);
    }

    
    @Override
    public <V extends Number> ADAM<V> create(Tsr<V> target) {
        return new ADAM<>(0, _learningRate, target);
    }

    public <V extends Number> ADAM<V> create(Tsr<V> momentum, Tsr<V> velocity) {
        return new ADAM<>(0, _learningRate, momentum, velocity);
    }

}
