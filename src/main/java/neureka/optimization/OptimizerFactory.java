package neureka.optimization;

import neureka.Tsr;

public interface OptimizerFactory {

    <V extends Number> Optimizer<V> create(Tsr<V> target);

}
