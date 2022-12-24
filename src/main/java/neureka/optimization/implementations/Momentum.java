package neureka.optimization.implementations;

import neureka.Tsr;
import neureka.common.utility.LogUtil;
import neureka.optimization.Optimizer;

import java.util.List;

public class Momentum<V extends Number> implements Optimizer<V>
{
    private final double lr; // learning rate
    private final double decay; // decay rate
    private final Tsr<Number> v; // velocity:

    Momentum( Tsr<V> target, double learningRate, double decay ) {
        LogUtil.nullArgCheck( target, "target", Tsr.class );
        List<Integer> shape = target.shape();
        v = Tsr.of(target.getItemType(), shape, 0).getMut().upcast(Number.class);
        lr = learningRate; // Step size/learning rate is 0.01 by default!
        this.decay = decay; // Decay rate is 0.9 by default!
    }

    @Override
    public Tsr<V> optimize( Tsr<V> w ) {
        LogUtil.nullArgCheck( w, "w", Tsr.class ); // The input must not be null!
        Tsr<Number> g = w.gradient().get().mut().upcast(Number.class);
        v.getMut().timesAssign(decay);
        v.getMut().plusAssign(g.times(1 - decay));
        return Tsr.of("-" + lr + " * I[0]", (Tsr<V>) v);
    }

}
