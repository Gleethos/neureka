package neureka.optimization.implementations;

import neureka.Tsr;
import neureka.common.utility.LogUtil;
import neureka.optimization.Optimizer;

import java.util.List;

/**
 * Adaptive Gradients, or AdaGrad for short, is an extension of the gradient descent optimization
 * algorithm that adjusts the step size for each parameter based on the squared gradients
 * seen over the course of previous optimization steps.
 *
 * @param <V> The super type of the value item type for the tensors whose gradients can be optimized by this.
 */
public class AdaGrad<V extends Number> implements Optimizer<V>
{
    private static final double E = 1e-8;

    private final double lr; // learning rate
    private final Tsr<Number> h; // sum of squared gradients:


    public AdaGrad( Tsr<V> target ) {
        LogUtil.nullArgCheck( target, "target", Tsr.class );
        List<Integer> shape = target.shape();
        h = Tsr.of(target.getItemType(), shape, 0).getMut().upcast(Number.class);
        lr = 0.01; // Step size/learning rate is 0.01 by default!
    }

    @Override
    public Tsr<V> optimize( Tsr<V> w ) {
        LogUtil.nullArgCheck( w, "w", Tsr.class ); // The input must not be null!
        Tsr<Number> g = w.gradient().get().mut().upcast(Number.class);
        h.getMut().plusAssign(g.power(2));
        return Tsr.of("-"+ lr +" * ", g, " / ( ( ", h, " ** 0.5 ) + "+E+" )");
    }

}