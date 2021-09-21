package neureka.backend.api.algorithms.fun;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.RecursiveExecutor;
import neureka.calculus.implementations.FunctionNode;
import neureka.devices.Device;

/**
 *  The {@link ExecutionDispatcher} lambda
 *  is the final execution procedure which is responsible for electing an {@link neureka.backend.api.ImplementationFor}
 *  the chosen {@link Device} in a given {@link ExecutionCall}.
 *  However the  {@link ExecutionDispatcher} does not have to select a device specific implementation.
 *  It can also occupy the rest of the execution without any other steps being taken.
 *  For example, a {@link neureka.backend.api.ImplementationFor} or a {@link RecursiveExecutor}
 *  would not be used if not explicitly called.
 *  Bypassing other procedures is useful for full control and of course to implement unorthodox types of operations
 *  like the {@link neureka.backend.standard.operations.other.Reshape} operation
 *  which is very different from classical operations.
 *  Although the {@link ExecutionCall} passed to implementations of this will contain
 *  a fairly suitable {@link Device} assigned to a given {@link neureka.backend.api.Algorithm},
 *  one can simply ignore it and find a custom one which fits the contents of the given
 *  {@link ExecutionCall} instance better.
 */
public interface ExecutionDispatcher {

    Tsr<?> dispatch( FunctionNode caller, ExecutionCall<? extends Device<?>> call );

}
