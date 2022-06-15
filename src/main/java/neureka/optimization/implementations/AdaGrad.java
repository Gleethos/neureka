package neureka.optimization.implementations;

import neureka.Tsr;
import neureka.common.utility.LogUtil;
import neureka.optimization.Optimizer;

public class AdaGrad<V> implements Optimizer<V>
{
    private final double a; // learning rate
    private final Tsr<V> h; // sum of squared gradients:


    public AdaGrad(Tsr<V> target) {
        LogUtil.nullArgCheck( target, "target", Tsr.class );
        int[] shape = target.getNDConf().shape();
        h = Tsr.of(target.getValueClass(), shape, 0);
        a  = 0.01; // Step size/learning rate is 0.01 by default!
    }

    @Override
    public Tsr<V> optimize( Tsr<V> w ) {
        LogUtil.nullArgCheck( w, "w", Tsr.class ); // The input must not be null!
        Tsr<V> g = w.getGradient();
        h.plusAssign(g.power(2));
        return Tsr.of("-"+a+" * ", g, " / ( ( ", h, " ^ 0.5 ) + 1e-8 )");
    }

}