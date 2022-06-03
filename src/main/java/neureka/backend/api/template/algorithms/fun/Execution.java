package neureka.backend.api.template.algorithms.fun;

import neureka.backend.api.ExecutionCall;
import neureka.calculus.Function;
import neureka.devices.Device;

/**
 *  Implementations of this functional interface
 *  is supposed to be the final execution procedure responsible for dispatching
 *  the execution further into the backend.
 *  This usually involves electing an {@link neureka.backend.api.ImplementationFor}
 *  the chosen {@link Device} in a given {@link ExecutionCall}.
 *  However, the  {@link Execution} does not have to select a device specific implementation.
 *  It can also occupy the rest of the execution without any other steps being taken.
 *  For example, a {@link neureka.backend.api.ImplementationFor} or a {@link neureka.calculus.internal.RecursiveExecutor}
 *  would not be used if not explicitly called.
 *  Bypassing other procedures is useful for full control and of course to implement unorthodox types of operations
 *  like the {@link neureka.backend.main.operations.other.Reshape} operation
 *  which is very different from classical operations.
 *  Although the `ExecutionCall` passed to implementations of this will contain
 *  a fairly suitable `Device` assigned to a given `neureka.backend.api.Algorithm`,
 *  one can simply ignore it and find a custom one which fits the contents of the given
 *  {@link ExecutionCall} instance better.
 */
public interface Execution {

    /**
     * @param caller The caller {@link Function} from which the request for execution originated.
     * @param call The {@link ExecutionCall} which should be executed.
     * @return A {@link Result} instance wrapping a {@link neureka.Tsr} and optionally also an {@link ADAgentSupplier}.
     */
    Result execute( Function caller, ExecutionCall<? extends Device<?>> call );

}
