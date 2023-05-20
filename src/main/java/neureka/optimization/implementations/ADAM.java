/*
MIT License

Copyright (c) 2019 Gleethos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

           _____          __  __
     /\   |  __ \   /\   |  \/  |
    /  \  | |  | | /  \  | \  / |
   / /\ \ | |  | |/ /\ \ | |\/| |       Adaptive - Moment - Estimation
  / ____ \| |__| / ____ \| |  | |
 /_/    \_\_____/_/    \_\_|  |_|

    A tensor gradient optimizer.

*/

package neureka.optimization.implementations;

import neureka.Tensor;
import neureka.common.utility.LogUtil;
import neureka.optimization.Optimizer;

/**
 *  ADAM (short for Adaptive Moment Estimation) is an adaptive learning rate optimization algorithm that utilises both
 *  momentum and scaling, combining the benefits of RMSProp and SGD with respect to Momentum.
 *  The optimizer is designed to be appropriate for non-stationary
 *  objectives and problems with very noisy and/or sparse gradients.
 *
 * @param <V> The value type parameter of the tensor whose gradients are being optimized.
 */
public final class ADAM<V extends Number> implements Optimizer<V>
{
    // Constants:
    private static final double B1 = 0.9;
    private static final double B2 = 0.999;
    private static final double E = 1e-8;

    // Parameter
    private final double lr; // learning rate

    // Variables:
    private Tensor<V> m; // momentum
    private Tensor<V> v; // velocity
    private long t; // time

    ADAM( long t, double lr, Tensor<V> target ) {
        this(t, lr, // Step size/learning rate is 0.01 by default!
            Tensor.of(target.getItemType(), target.shape(), 0), // momentum
            Tensor.of(target.getItemType(), target.shape(), 0)  // velocity
        );
    }

    ADAM(long t, double lr, Tensor<V> m, Tensor<V> v ) {
        LogUtil.nullArgCheck( m, "m", Tensor.class );
        LogUtil.nullArgCheck( v, "v", Tensor.class );
        this.m = m;
        this.v = v;
        this.t = t;
        this.lr = lr;
    }

    @Override
    public Tensor<V> optimize(Tensor<V> w ) {
        LogUtil.nullArgCheck( w, "w", Tensor.class ); // The input must not be null!
        t++;
        Tensor<V> g = w.gradient().orElseThrow( () -> new IllegalStateException("Gradient missing! Cannot perform optimization.") );
        double b1Inverse = ( 1 - B1 );
        double b2Inverse = ( 1 - B2 );
        double b1hat = ( 1 - Math.pow( B1, t ) );
        double b2hat = ( 1 - Math.pow( B2, t ) );
        m = Tensor.of(B1+" * ", m, " + "+b1Inverse+" * ", g);
        v = Tensor.of(B2+" * ", v, " + "+b2Inverse+" * (", g,"**2 )");
        Tensor<V> mh = Tensor.of(m, "/"+b1hat);
        Tensor<V> vh = Tensor.of(v, "/"+b2hat);
        Tensor<V> newg = Tensor.of("-"+ lr +" * ",mh," / (",vh,"**0.5 + "+E+")");
        mh.getMut().delete();
        vh.getMut().delete();
        return newg;
    }

    public final Tensor<V> getMomentum() { return m; }

    public final Tensor<V> getVelocity() { return v; }

    public final long getTime() { return t; }

    public final double getLearningRate() { return lr; }

}
