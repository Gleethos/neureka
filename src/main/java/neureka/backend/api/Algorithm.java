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


           _                  _ _   _
     /\   | |                (_) | | |
    /  \  | | __ _  ___  _ __ _| |_| |__  _ __ ___
   / /\ \ | |/ _` |/ _ \| '__| | __| '_ \| '_ ` _ \
  / ____ \| | (_| | (_) | |  | | |_| | | | | | | | |
 /_/    \_\_|\__, |\___/|_|  |_|\__|_| |_|_| |_| |_|
              __/ |
             |___/

------------------------------------------------------------------------------------------------------------------------
*/


package neureka.backend.api;

import neureka.Tsr;
import neureka.backend.api.algorithms.fun.*;
import neureka.backend.standard.algorithms.FunAlgorithm;
import neureka.devices.Device;


/**
 *   This class is the middle layer of the 3 tier abstraction architecture of this backend. <br>
 *
 *   Conceptually an implementation of the {@link Algorithm} interface represents "a sub-kind of operation" for
 *   an instance of an implementation of the {@link Operation} interface. <br>
 *   The "+" operator for example has different {@link Algorithm} instances tailored to specific requirements
 *   originating from different {@link ExecutionCall} instances with unique arguments.
 *   {@link Tsr} instances within an execution call having the same shape would
 *   cause the {@link Operation} instance to chose an {@link Algorithm} instance which is responsible
 *   for performing element-wise operations, whereas otherwise the {@link neureka.backend.standard.algorithms.Broadcast}
 *   algorithm might be called to perform the operation.
 */
public interface Algorithm<C extends Algorithm<C>>
extends SuitabilityPredicate, ForwardADPredicate, BackwardADPredicate, ADAgentSupplier, ExecutionPreparation, ExecutionDispatcher
{
    /**
     * This is a factory method for creating a new instance of this {@link FunAlgorithm} class.
     *
     * @param name The name of the functional algorithm.
     * @return An new {@link FunAlgorithm} with the provided name.
     */
    static FunAlgorithm withName(String name) {
        return new FunAlgorithm(name);
    }

    /**
     *  The name of an {@link Algorithm} may be used for OpenCL kernel compilation or simply
     *  for debugging purposes to identify which type of algorithm is being executed at any
     *  given time...
     *
     * @return The name of this {@link Algorithm}.
     */
    String getName();

    //---

    /**
     * Implementations of the Algorithm interface ought to express a compositional design pattern. <br>
     * This means that concrete implementations of an algorithm for a device are not extending
     * an Algorithm, they are components of it instead. <br>
     * These components can be stored on an Algorithm by passing
     * a Device class as key and an ImplementationFor instance as value.
     *
     * @param deviceClass    The class of the {@link Device} for which an implementation should be set.
     * @param implementation The {@link ImplementationFor} the provided {@link Device} type.
     * @param <D>            The type parameter of the {@link Device} type for which
     *                       an implementation should be set in this {@link Device}.
     * @param <I>            The type of the {@link ImplementationFor} the provided {@link Device} type.
     * @return This very {@link Algorithm} instance to allow for method chaining.
     */
    <D extends Device<?>, I extends ImplementationFor<D>> C setImplementationFor(Class<D> deviceClass, I implementation );

    /**
     * An {@link ImplementationFor} a specific {@link Device} can be accessed by passing the class of
     * the {@link Device} for which an implementation should be returned.
     * An Algorithm instance ought to contain a collection of these {@link Device} specific
     * implementations...
     *
     * @param deviceClass The class of the device for which the stored algorithm implementation should be returned.
     * @param <D>         The type parameter which has to be a class extending the Device interface.
     * @return The implementation for the passed device type class.
     */
    <D extends Device<?>> ImplementationFor<D> getImplementationFor( Class<D> deviceClass );

    /**
     * An {@link ImplementationFor} a specific {@link Device} can be accessed by passing the
     * the {@link Device} for which an implementation should be returned.
     * An Algorithm instance ought to contain a collection of these {@link Device} specific
     * implementations...
     *
     * @param device The device for which the stored algorithm implementation should be returned.
     * @param <D>    type parameter which has to be a class extending the Device interface.
     * @return The implementation for the passed device type class.
     */
    default <D extends Device<?>> ImplementationFor<D> getImplementationFor( D device ) {
        return (ImplementationFor<D>) getImplementationFor(device.getClass());
    }


}
