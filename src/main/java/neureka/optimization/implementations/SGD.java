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

    _____  _____ _____
   / ____|/ ____|  __ \
  | (___ | |  __| |  | |
   \___ \| | |_ | |  | |         Stochastic - Gradient - Descent
   ____) | |__| | |__| |
  |_____/ \_____|_____/

    A tensor gradient optimizer.

*/

package neureka.optimization.implementations;

import neureka.Neureka;
import neureka.Tsr;
import neureka.calculus.Function;
import neureka.optimization.Optimizer;

public class SGD<V> implements Optimizer<V>
{
    private final double _learningRate;
    private final Function _function;

    public SGD( double leaningRate )
    {
        _learningRate = leaningRate;
        _function = Function.of("I[ 0 ] <- (-1 * (I[ 0 ] - "+leaningRate+"))", false);
    }

    @Override
    public Tsr<V> optimize(Tsr<V> w ) {
        Tsr<V> g = w.getGradient();
        return Neureka.get().context().getFunction().idy().call( _function.call( g ) );
    }

    public double learningRate() {
        return _learningRate;
    }

    @Override
    public boolean update( OwnerChangeRequest<Tsr<V>> changeRequest ) {
        changeRequest.executeChange();
        return true;
    }
}
