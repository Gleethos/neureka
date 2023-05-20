package neureka.optimization.implementations;

import neureka.Shape;
import neureka.Tensor;
import neureka.common.utility.LogUtil;
import neureka.optimization.Optimizer;

public class Momentum<V extends Number> implements Optimizer<V>
{
    private final double lr; // learning rate
    private final double decay; // decay rate
    private final Tensor<Number> v; // velocity:

    Momentum(Tensor<V> target, double learningRate, double decay ) {
        LogUtil.nullArgCheck( target, "target", Tensor.class );
        Shape shape = target.shape();
        v = Tensor.of(target.getItemType(), shape, 0).getMut().upcast(Number.class);
        lr = learningRate; // Step size/learning rate is 0.01 by default!
        this.decay = decay; // Decay rate is 0.9 by default!
    }

    @Override
    public Tensor<V> optimize(Tensor<V> w ) {
        LogUtil.nullArgCheck( w, "w", Tensor.class ); // The input must not be null!
        Tensor<Number> g = w.gradient().get().mut().upcast(Number.class);
        v.getMut().timesAssign(decay);
        v.getMut().plusAssign(g.times(1 - decay));
        return Tensor.of("-" + lr + " * I[0]", (Tensor<V>) v);
    }

}
