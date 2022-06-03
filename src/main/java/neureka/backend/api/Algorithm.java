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
import neureka.backend.api.template.algorithms.fun.ADSupportPredicate;
import neureka.backend.api.template.algorithms.fun.Execution;
import neureka.backend.api.template.algorithms.fun.SuitabilityPredicate;


/**
 *   This class is the middle layer of the 3 tier compositional abstraction architecture of this backend, which
 *   consists of {@link Operation}s, {@link Algorithm}s and {@link ImplementationFor}. <br>
 *
 *   Conceptually an implementation of the {@link Algorithm} interface represents "a sub-kind of operation" for
 *   an instance of an implementation of the {@link Operation} interface. <br>
 *   The "+" operator for example has different {@link Algorithm} instances tailored to specific requirements
 *   originating from different {@link ExecutionCall} instances with unique arguments.
 *   {@link Tsr} instances within an execution call having the same shape would
 *   cause the {@link Operation} instance to choose an {@link Algorithm} instance which is responsible
 *   for performing element-wise operations, whereas otherwise the {@link neureka.backend.main.algorithms.Broadcast}
 *   algorithm might be called to perform the operation.
 */
public interface Algorithm
extends SuitabilityPredicate, ADSupportPredicate, Execution
{
    /**
     *  The name of an {@link Algorithm} may be used for OpenCL kernel compilation or simply
     *  for debugging purposes to identify which type of algorithm is being executed at any
     *  given time...
     *
     * @return The name of this {@link Algorithm}.
     */
    String getName();

}
