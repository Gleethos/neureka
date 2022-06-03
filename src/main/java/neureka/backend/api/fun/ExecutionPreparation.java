package neureka.backend.api.fun;

import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

/**
 *  An {@link Algorithm} will typically produce a result when executing an {@link ExecutionCall}.
 *  This result must be created somehow.
 *  A {@link ExecutionPreparation} implementation instance will do just that...
 *  Often times the first entry in the array of tensors stored inside the call
 *  will be null to serve as a position for the output to be placed at.
 *  The creation of this output tensor is of course highly dependent on the type
 *  of operation and algorithm that is currently being used.
 *  Element-wise operations for example will require the creation of an output tensor
 *  with the shape of the provided input tensors, whereas the execution of a
 *  linear operation like for example a broadcast operation will require a very different approach...
 */
 @FunctionalInterface
public interface ExecutionPreparation {

    /**
     * @param call The execution call which needs to be prepared for execution.
     * @return The prepared {@link ExecutionCall} instance.
     */
    ExecutionCall<? extends Device<?>> prepare(ExecutionCall<? extends Device<?>> call );

}
