package neureka.optimization.implementations;

import neureka.Shape;
import neureka.Tensor;
import neureka.common.utility.LogUtil;
import neureka.optimization.Optimizer;

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
    private final Tensor<Number> h; // sum of squared gradients:

    AdaGrad(Tensor<V> target, double learningRate ) {
        LogUtil.nullArgCheck( target, "target", Tensor.class );
        Shape shape = target.shape();
        h = Tensor.of(target.getItemType(), shape, 0).getMut().upcast(Number.class);
        lr = learningRate; // Step size/learning rate is 0.01 by default!
    }

    @Override
    public Tensor<V> optimize(Tensor<V> w ) {
        LogUtil.nullArgCheck( w, "w", Tensor.class ); // The input must not be null!
        Tensor<Number> g = w.gradient().get().mut().upcast(Number.class);
        h.getMut().plusAssign(g.power(2));
        return Tensor.of("-"+ lr +" * ", g, " / ( ( ", h, " ** 0.5 ) + "+E+" )");
    }

}