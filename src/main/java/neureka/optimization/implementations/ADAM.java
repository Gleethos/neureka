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

import neureka.Tsr;
import neureka.common.utility.LogUtil;
import neureka.optimization.Optimizer;

import java.util.List;

/**
 *  ADAM (short for Adaptive Moment Estimation) is an adaptive learning rate optimization algorithm that utilises both
 *  momentum and scaling, combining the benefits of RMSProp and SGD with respect to Momentum.
 *  The optimizer is designed to be appropriate for non-stationary
 *  objectives and problems with very noisy and/or sparse gradients.
 *
 * @param <V> The value type parameter of the tensor whose gradients are being optimized.
 */
public class ADAM<V extends Number> implements Optimizer<V>
{
    // Constants:
    private static final double B1 = 0.9;
    private static final double B2 = 0.999;
    private static final double E = 1e-8;

    // Parameter
    private final double lr; // learning rate

    // Variables:
    private Tsr<V> m; // momentum
    private Tsr<V> v; // velocity
    private long t = 0; // time

    public ADAM(Tsr<V> target) {
        LogUtil.nullArgCheck( target, "target", Tsr.class );
        List<Integer> shape = target.shape();
        m  = Tsr.of(target.getValueClass(), shape, 0);
        v  = Tsr.of(target.getValueClass(), shape, 0);
        lr = 0.01; // Step size/learning rate is 0.01 by default!
    }

    @Override
    public Tsr<V> optimize( Tsr<V> w ) {
        LogUtil.nullArgCheck( w, "w", Tsr.class ); // The input must not be null!
        t++;
        Tsr<V> g = w.getGradient();
        double b1Inverse = ( 1 - B1 );
        double b2Inverse = ( 1 - B2 );
        double b1hat = ( 1 - Math.pow( B1, t ) );
        double b2hat = ( 1 - Math.pow( B2, t ) );
        m = Tsr.of(B1+" * ", m, " + "+b1Inverse+" * ", g);
        v = Tsr.of(B2+" * ", v, " + "+b2Inverse+" * (", g,"^2 )");
        Tsr<V> mh = Tsr.of(m, "/"+b1hat);
        Tsr<V> vh = Tsr.of(v, "/"+b2hat);
        Tsr<V> newg = Tsr.of("-"+ lr +" * ",mh," / (",vh,"^0.5 + "+E+")");
        mh.getUnsafe().delete();
        vh.getUnsafe().delete();
        return newg;
    }

}
