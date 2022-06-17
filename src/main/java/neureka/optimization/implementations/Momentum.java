package neureka.optimization.implementations;

import neureka.Tsr;
import neureka.common.utility.LogUtil;
import neureka.optimization.Optimizer;

public class Momentum<V extends Number> implements Optimizer<V>
{
    private final double lr; // learning rate
    private final double decay; // decay rate
    private final Tsr<Number> v; // velocity:

    public Momentum( Tsr<V> target ) {
        LogUtil.nullArgCheck( target, "target", Tsr.class );
        int[] shape = target.getNDConf().shape();
        v = Tsr.of(target.getValueClass(), shape, 0).getUnsafe().upcast(Number.class);
        lr = 0.01; // Step size/learning rate is 0.01 by default!
        decay = 0.9; // Decay rate is 0.9 by default!
    }

    @Override
    public Tsr<V> optimize( Tsr<V> w ) {
        LogUtil.nullArgCheck( w, "w", Tsr.class ); // The input must not be null!
        Tsr<Number> g = w.getGradient().getUnsafe().upcast(Number.class);
        v.timesAssign(decay);
        v.plusAssign(g.times(1 - decay));
        return Tsr.of("-" + lr + " * I[0]", (Tsr<V>) v);
    }

}
