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

/**
 *  {@link ADAgent} stands for "Auto-Differentiation-Agent", meaning
 *  that implementations of this class are responsible for managing
 *  forward- and reverse- mode differentiation actions.
 *  These actions are accessible through the "{@link ADAgent#forward(GraphNode, Tsr)}"
 *  and "{@link ADAgent#backward(GraphNode, Tsr)}" method which are being triggered
 *  by instances of the {@link GraphNode} class during propagation.
 */
public interface ADAgent
{
    /**
     *  The auto-differentiation forward pass of an ADAgent
     *  propagate partial differentiations forward along the computation graph.
     *
     * @param target The node which is targeted to hold the partial derivative.
     * @param derivative The partial derivative which ought to be propagated forward.
     * @param <T> The type argument of the tensor that is being used.
     * @return The result of a forward mode auto differentiation.
     */
    <T> Tsr<T> forward( GraphNode<T> target, Tsr<T> derivative );

    /**
     *  The auto-differentiation backward pass of an ADAgent
     *  propagates error values backward along the computation graph.
     *
     * @param target The node which is targeted to hold the new error.
     * @param error The error which ought to be propagated backward.
     * @param <T> The type argument of the tensor that is being used.
     * @return The result of a backward mode auto differentiation, and new error for the targeted graph node.
     */
    <T> Tsr<T> backward( GraphNode<T> target, Tsr<T> error );

    /**
     * @return A partial derivative or null if the agent does not contain a partial derivative...
     */
    Tsr<?> derivative();

    boolean hasForward();

    boolean hasBackward();

    String toString();

}
