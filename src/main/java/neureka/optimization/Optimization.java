package neureka.optimization;

import neureka.Tensor;

public interface Optimization<V> {

    Tensor<V> optimize(Tensor<V> w );

}
