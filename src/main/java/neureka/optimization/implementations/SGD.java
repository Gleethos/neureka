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

import neureka.Tsr;
import neureka.common.utility.LogUtil;
import neureka.optimization.Optimizer;

/**
 * Stochastic Gradient Descent is an iterative optimization technique
 * that uses the gradient of a weight variable to adjust said variable,
 * in order to reduce the error used to calculate said gradient.
 * This simple stochastic gradient descent algorithm does not
 * optimize the gradient based on previous gradients (network forward and backward passes)
 * but simply applies the gradient value based on each example within
 * the training dataset.
 *
 * @param <V> The value type parameter of the tensor whose gradients are being optimized.
 */
public class SGD<V> implements Optimizer<V>
{
    private final double _lr; // learning rate

    public SGD( double learningRate ) {
        _lr = learningRate; // Step size/learning rate is 0.01 by default!
    }

    @Override
    public Tsr<V> optimize( Tsr<V> w ) {
        LogUtil.nullArgCheck( w, "w", Tsr.class ); // The input must not be null!
        Tsr<V> g = w.gradient().orElseThrow(()->new IllegalStateException("Gradient missing!"));
        return Tsr.of("-" + _lr + " * i0", g);
    }

    public double learningRate() { return _lr; }

}
