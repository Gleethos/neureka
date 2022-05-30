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

import neureka.Neureka;
import neureka.Tsr;
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
public class ADAM<V> implements Optimizer<V> {

    // Constants:
    private final Tsr<V> a;
    private final Tsr<V> b1;
    private final Tsr<V> b2;
    private final Tsr<V> e;

    // Variables:
    private Tsr<V> m; // Momentum
    private Tsr<V> v; // Velocity

    public ADAM(Tsr<V> target) {
        LogUtil.nullArgCheck( target, "target", Tsr.class );
        int[] shape = target.getNDConf().shape();
        m  = Tsr.of(target.getValueClass(), shape, 0);
        v  = Tsr.of(target.getValueClass(), shape, 0);
        a  = Tsr.of(target.getValueClass(), shape, 0.01); // Step size!
        b1 = Tsr.of(target.getValueClass(), shape, 0.9);
        b2 = Tsr.of(target.getValueClass(), shape, 0.999);
        e  = Tsr.of(target.getValueClass(), shape, 1e-7);
    }

    private Tsr<V> _optimize( Tsr<V> w ) {
        Tsr<V> g = w.getGradient();
        m = Tsr.of(b1, "*", m, " + ( 1-", b1, ") *", g);
        v = Tsr.of(b2, "*", v, " + ( 1-", b2, ") * (", g,"^2 )");
        Tsr<V> mh = Tsr.of(m, "/(1-", b1, ")");
        Tsr<V> vh = Tsr.of(v, "/(1-", b2, ")");
        Tsr<V> newg = Tsr.of("-",a,"*",mh,"/(",vh,"^0.5+",e,")");
        Neureka.get().backend().getFunction().idy().call(g, newg);
        mh.getUnsafe().delete();
        vh.getUnsafe().delete();
        newg.getUnsafe().delete();
        return g;
    }

    @Override
    public Tsr<V> optimize( Tsr<V> w ) {
        LogUtil.nullArgCheck( w, "w", Tsr.class );
        return _optimize(w);
    }

    @Override
    public boolean update( OwnerChangeRequest<Tsr<V>> changeRequest ) {
        changeRequest.executeChange(); // This can be an 'add', 'remove' or 'transfer' of this component!
        return true;
    }
}
