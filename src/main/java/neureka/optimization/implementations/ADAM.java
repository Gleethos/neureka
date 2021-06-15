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
import neureka.optimization.Optimizer;

public class ADAM<ValType> implements Optimizer<ValType> {

    // VARIABLES...
    private final Tsr<ValType> a;
    private final Tsr<ValType> b1;
    private final Tsr<ValType> b2;
    private final Tsr<ValType> e;
    Tsr<ValType> m;
    Tsr<ValType> v;

    ADAM(Tsr<ValType> target) {
        int[] shape = target.getNDConf().shape();
        m  = (Tsr<ValType>) Tsr.of(shape, 0);
        v  = (Tsr<ValType>) Tsr.of(shape, 0);
        a  = (Tsr<ValType>) Tsr.of(shape, 0.01); // Step size!
        b1 = (Tsr<ValType>) Tsr.of(shape, 0.9);
        b2 = (Tsr<ValType>) Tsr.of(shape, 0.999);
        e  = (Tsr<ValType>) Tsr.of(shape, 1e-7);
    }

    private void _optimize(Tsr<ValType> w) {
        Tsr<ValType> g = w.getGradient();
        m = (Tsr<ValType>) Tsr.of(b1, "*", m, " + ( 1-", b1, ") *", g);
        v = (Tsr<ValType>) Tsr.of(b2, "*", v, " + ( 1-", b2, ") * (", g,"^2 )");
        Tsr<ValType> mh = (Tsr<ValType>) Tsr.of(m, "/(1-", b1, ")");
        Tsr<ValType> vh = (Tsr<ValType>) Tsr.of(v, "/(1-", b2, ")");
        Tsr<ValType> newg = (Tsr<ValType>) Tsr.of("-",a,"*",mh,"/(",vh,"^0.5+",e,")");
        Neureka.get().context().getFunction().idy().call(new Tsr[]{g, newg});
    }

    @Override
    public void optimize(Tsr<ValType> t) {
        _optimize(t);
    }

    @Override
    public void update(Tsr<ValType> oldOwner, Tsr<ValType> newOwner) {
        
    }
}
