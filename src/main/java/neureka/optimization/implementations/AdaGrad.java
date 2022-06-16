package neureka.optimization.implementations;

import neureka.Tsr;
import neureka.common.utility.LogUtil;
import neureka.optimization.Optimizer;

public class AdaGrad<V extends Number> implements Optimizer<V>
{
    private static final double E = 1e-8;

    private final double lr; // learning rate
    private final Tsr<Number> h; // sum of squared gradients:


    public AdaGrad( Tsr<V> target ) {
        LogUtil.nullArgCheck( target, "target", Tsr.class );
        int[] shape = target.getNDConf().shape();
        h = Tsr.of(target.getValueClass(), shape, 0).getUnsafe().upcast(Number.class);
        lr = 0.01; // Step size/learning rate is 0.01 by default!
    }

    @Override
    public Tsr<V> optimize( Tsr<V> w ) {
        LogUtil.nullArgCheck( w, "w", Tsr.class ); // The input must not be null!
        Tsr<Number> g = w.getGradient().getUnsafe().upcast(Number.class);
        h.plusAssign(g.power(2));
        return Tsr.of("-"+ lr +" * ", g, " / ( ( ", h, " ^ 0.5 ) + "+E+" )");
    }

}