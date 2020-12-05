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

            _         _                  _   _____             _
      /\   | |       | |                | | |  __ \           (_)
     /  \  | |__  ___| |_ _ __ __ _  ___| |_| |  | | _____   ___  ___ ___
    / /\ \ | '_ \/ __| __| '__/ _` |/ __| __| |  | |/ _ \ \ / / |/ __/ _ \
   / ____ \| |_) \__ \ |_| | | (_| | (__| |_| |__| |  __/\ V /| | (_|  __/
  /_/    \_\_.__/|___/\__|_|  \__,_|\___|\__|_____/ \___| \_/ |_|\___\___|


*/

package neureka.devices;

import neureka.Tsr;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.operations.OperationType;
import neureka.calculus.backend.implementations.OperationTypeImplementation;

import java.lang.ref.Cleaner;

/**
 *  The is the abstract precursor class providing
 *  some useful implementations for core concepts which are most likely
 *  applicable to most concrete implementations of the Device interface.
 *  These class provides the following features :
 *
 *  - A Cleaner instance used for freeing resources of the device.
 *
 *  - An component update implementations which simply calls the swap method of the device.
 *
 *  - An implementation for the execution method which calls the underlying calculus backend.
 *
 * @param <ValueType>
 */
public abstract class AbstractDevice<ValueType> extends AbstractBaseDevice<ValueType>
{
    private static final Cleaner _CLEANER = Cleaner.create();

    /**
     *  This method is the internal execution routine called by it's public counterpart
     *  and implemented by classes extending this very abstract class.
     *  It substitutes the implementation of this public "execute" method
     *  in order to make any execution call on any device extending this class
     *  checked before execution.
     *  The checking occurs in the public "execute" method of this class.
     *
     * @param tensors An array of input tensors.
     * @param d The index of the input which ought to be derived.
     * @param type The type of operation.
     */
    protected abstract void _execute( Tsr[] tensors, int d, OperationType type );

    @Override
    public void update( Tsr oldOwner, Tsr newOwner ) {
        swap( oldOwner, newOwner );
    }

    @Override
    public Device cleaning( Tsr tensor, Runnable action ) {
        _cleaning(tensor, action);
        return this;
    }

    protected void _cleaning( Object o, Runnable action ) {
        _CLEANER.register(o, action);
    }

    @Override
    public Device<ValueType> execute( ExecutionCall call )
    {
        call = call.getImplementation().instantiateNewTensorsForExecutionIn(call);
        for ( Tsr<?> t : call.getTensors() ) {
            if ( t == null ) throw new IllegalArgumentException(
                    "Device arguments may not be null!\n" +
                            "One or more tensor arguments within the given ExecutionCall instance is null."
            );
        }
        ( (OperationTypeImplementation<Object>) call.getImplementation() )
                .recursiveReductionOf(
                    call,
                    c -> _execute( c.getTensors(), c.getDerivativeIndex(), c.getType() )
                );
        return this;
    }

}
