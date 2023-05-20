package neureka.optimization;

import neureka.Tensor;

public interface OptimizerFactory {

    <V extends Number> Optimizer<V> create(Tensor<V> target);

}
