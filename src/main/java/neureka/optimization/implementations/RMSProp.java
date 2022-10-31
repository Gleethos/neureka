package neureka.optimization.implementations;

import neureka.Tsr;
import neureka.common.utility.LogUtil;
import neureka.optimization.Optimizer;

/**
 * Root Mean Squared Propagation, or RMSProp,
 * is an extension of gradient descent and the AdaGrad version of gradient
 * descent that uses a decaying average of partial gradients in the adaptation of the
 * step size for each parameter.
 * It is similar to {@link AdaGrad} in that it uses a moving average of
 * the squared gradients to scale the learning rate.
 *
 * @param <V> The super type of the value item type for the tensors whose gradients can be optimized by this.
 */
public class RMSProp<V extends Number> implements Optimizer<V>
{
    private final double lr; // learning rate
    private final double decay; // decay rate
    private final Tsr<Number> h; // sum of squared gradients:

    public RMSProp(Tsr<Number> target ) {
        LogUtil.nullArgCheck( target, "target", Tsr.class );
        h = Tsr.of(target.getItemType(), target.shape(), 0);
        lr = 0.001; // Step size/learning rate is 0.001 by default!
        decay = 0.9; // Decay rate is 0.9 by default!
    }

    @Override
    public Tsr<V> optimize( Tsr<V> w ) {
        LogUtil.nullArgCheck( w, "w", Tsr.class ); // The input must not be null!
        Tsr<Number> g = w.gradient().orElse(null).getMut().upcast(Number.class);
        h.getMut().timesAssign(decay);
        h.getMut().plusAssign(g.power(2).times(1 - decay));
        return Tsr.of("-" + lr + " * ", g, " / ( ( ", h, " ** 0.5 ) + 1e-8 )");
    }
}