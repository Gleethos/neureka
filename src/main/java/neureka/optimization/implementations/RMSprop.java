package neureka.optimization.implementations;

import neureka.Tsr;
import neureka.common.utility.LogUtil;
import neureka.optimization.Optimizer;

public class RMSprop<V extends Number> implements Optimizer<V>
{
    private final double lr; // learning rate
    private final double decay; // decay rate
    private final Tsr<Number> h; // sum of squared gradients:

    public RMSprop( Tsr<Number> target ) {
        LogUtil.nullArgCheck( target, "target", Tsr.class );
        h = Tsr.of(target.getValueClass(), target.getNDConf().shape(), 0);
        lr = 0.001; // Step size/learning rate is 0.001 by default!
        decay = 0.9; // Decay rate is 0.9 by default!
    }

    @Override
    public Tsr<V> optimize( Tsr<V> w ) {
        LogUtil.nullArgCheck( w, "w", Tsr.class ); // The input must not be null!
        Tsr<Number> g = w.getGradient().getUnsafe().upcast(Number.class);
        h.timesAssign(decay);
        h.plusAssign(g.power(2).times(1 - decay));
        return Tsr.of("-" + lr + " * ", g, " / ( ( ", h, " ^ 0.5 ) + 1e-8 )");
    }
}