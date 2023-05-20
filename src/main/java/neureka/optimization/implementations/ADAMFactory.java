package neureka.optimization.implementations;

import neureka.Tensor;
import neureka.optimization.OptimizerFactory;

public class ADAMFactory implements OptimizerFactory 
{
    private final double _learningRate;
    private final long _time;

    public ADAMFactory() { this(0.01, 0); }

    // The copy constructor should be private, use withers instead!
    private ADAMFactory( double learningRate, long time ) {
        if ( time < 0 ) throw new IllegalArgumentException("The time must be a positive number!");
        _learningRate = learningRate;
        _time = time;
    }
    
    // Withers:

    public ADAMFactory withLearningRate(double learningRate) { return new ADAMFactory(learningRate, _time); }

    public ADAMFactory withTime(long time) { return new ADAMFactory(_learningRate, time); }

    @Override
    public <V extends Number> ADAM<V> create(Tensor<V> target) {
        return new ADAM<>(_time, _learningRate, target);
    }

    public <V extends Number> ADAM<V> create(Tensor<V> momentum, Tensor<V> velocity) {
        return new ADAM<>(_time, _learningRate, momentum, velocity);
    }

}
