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

            _____                          _
      /\   |  __ \   /\                   | |
     /  \  | |  | | /  \   __ _  ___ _ __ | |_
    / /\ \ | |  | |/ /\ \ / _` |/ _ \ '_ \| __|
   / ____ \| |__| / ____ \ (_| |  __/ | | | |_
  /_/    \_\_____/_/    \_\__, |\___|_| |_|\__|
                           __/ |
                          |___/

    Instances of this class manage forward and backward actions for
    forward-mode differentiation and reverse-mode differentiation.



*/

package neureka.autograd;

import neureka.Tsr;

import java.util.Optional;

/**
 *  {@link ADAgent} stands for "Auto-Differentiation-Agent", meaning
 *  that implementations of this class are responsible for managing
 *  forward- and reverse- mode differentiation actions.
 *  These differentiation actions are performed through the "{@link ADAgent#act(ADTarget)}"
 *  method which are being called
 *  by instances of the {@link GraphNode} class during propagation.
 *  An {@link ADAgent} may also wrap and expose a partial derivative
 *  which may or may not be present for certain operations.
 *  Said derivative must be tracked and flagged as derivative by a {@link GraphNode}
 *  to make sure that it will not be deleted after a forward pass.
 */
public interface ADAgent
{
    static WithADAction of( Tsr<?> derivative ) { return DefaultADAgent.ofDerivative( derivative ); }

    static ADAgent withAD( ADAction action ) { return of( null ).withAD( action ); }

    /**
     *  The auto-differentiation forward or backward pass of an ADAgent
     *  propagate partial differentiations forward along the computation graph.
     *
     * @param target A wrapper for the node which is targeted to hold the partial derivative and the error.
     * @param <T> The type argument of the tensor that is being used.
     * @return The result of a forward or backward mode auto differentiation.
     */
    <T> Tsr<T> act( ADTarget<T> target );

    /**
     * @return An optional partial derivative which may not be present if the agent does not wrap a partial derivative...
     */
    Optional<Tsr<?>> partialDerivative();

}
