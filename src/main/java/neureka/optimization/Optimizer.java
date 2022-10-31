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

    ____        _   _           _
   / __ \      | | (_)         (_)
  | |  | |_ __ | |_ _ _ __ ___  _ _______ _ __
  | |  | | '_ \| __| | '_ ` _ \| |_  / _ \ '__|
  | |__| | |_) | |_| | | | | | | |/ /  __/ |
   \____/| .__/ \__|_|_| |_| |_|_/___\___|_|
         | |
         |_|

*/

package neureka.optimization;

import neureka.common.composition.Component;
import neureka.Tsr;

/**
 *  {@link Optimizer}s are tensor components which implement the {@link Optimization} (functional)
 *  interface applying various optimization algorithms to the gradients of tensors. <br>
 *  A simple usage example would be the following:
 *  <pre>{@code
 *      Optimizer o = new SGD(0.01); // 0.01 learning rate
 *      Tsr<Float> w = Tsr.of(0f);
 *      w.set(o);
 *  }</pre>
 *  <br>
 *  If you want to create custom {@link Optimizer} implementations in a functional style
 *  take a look at the following example:
 *  <pre>{@code
 *      Optimizer.of( t -> {
 *          Tsr<?> gradient = t.getGradient();
 *          // ... apply algorithm ...
 *      })
 *  }</pre>
 *  Or consider using the factory method below to process gradients directly:
 *  <pre>{@code
 *      Optimizer.ofGradient( gradient -> {
 *          // ... apply algorithm ...
 *      })
 *  }</pre>
 *
 *
 * @param <V> The value type parameter of the tensors processed by this optimizer.
 */
public interface Optimizer<V> extends Component<Tsr<V>>, Optimization<V>
{
    /**
     * @param o The {@link Optimization} lambda which receives a tensor for optimization.
     * @return An {@link Optimizer} which will process any passed tensor directly (see {@link #ofGradient(Optimization)} which processes gradients).
     * @param <T> The value type parameter of the tensors processed by this optimizer.
     */
    static <T> Optimizer<T> of( Optimization<T> o ) {
        return new Optimizer<T>() {
            @Override public boolean update( OwnerChangeRequest<Tsr<T>> changeRequest ) { return true; }
            @Override public Tsr<T> optimize( Tsr<T> w ) { return o.optimize(w); }
        };
    }

    /**
     * @param o The {@link Optimization} lambda which receives the gradient of a tensor for optimization.
     * @return An {@link Optimizer} which will process the gradient of any passed tensor (see {@link #of(Optimization)} which processes tensors directly).
     * @param <T> The value type parameter of the tensors processed by this optimizer.
     */
    static <T> Optimizer<T> ofGradient( Optimization<T> o ) {
        return new Optimizer<T>() {
            @Override public boolean update( OwnerChangeRequest<Tsr<T>> changeRequest ) { return true; }
            @Override public Tsr<T> optimize( Tsr<T> w ) { return o.optimize(w.getGradientOrNull()); }
        };
    }

}
