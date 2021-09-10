package neureka.backend.api.algorithms.fun;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.implementations.FunctionNode;
import neureka.devices.Device;

/**
 *  The {@link ExecutionOrchestration} lambda
 *  is simply a bypass procedure which if provided will simply occupy
 *  the rest of the execution without any other steps being taken.
 *  For example, a {@link neureka.backend.api.ImplementationFor} or a {@link RecursiveExecutor}
 *  would not be used if not explicitly called.
 *  This bypassing is useful for full control and of course to implement unorthodox types of operations
 *  like the {@link neureka.backend.standard.operations.other.Reshape} operation
 *  which is very different from classical operations.
 */
public interface ExecutionOrchestration {

    Tsr<?> execute( FunctionNode caller, ExecutionCall<? extends Device<?>> call );

}
