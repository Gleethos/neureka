package neureka.calculus.backend.executions;

import neureka.device.Device;
import neureka.calculus.backend.ExecutionCall;

/**
 * This interface describes the functionality of an implementation
 * of an execution procedure for a specific Device (interface) instance
 * and OperationTypeImplementation (interface) instance!
 *
 * An instance of this interface is then a component of an OperationTypeImplementation instance
 * which is itself a component of the OperationType class.
 *
 * @param <TargetDevice> The Device type for which an implementation of this interface has been made.
 */
public interface ExecutorFor< TargetDevice extends Device >
{
    /**
     * Every ExecutorFor &lt; Device &gt; implementation needs to also
     * implement a lambda defined by the interface below.
     * The lambda shall take the call arguments and call
     * the specific methods of the Device type implementation
     * in order to satisfy the OperationTypeImplementation to which
     * this class belongs.
     *
     * @param <TargetDevice>
     */
    interface ExecutionOn< TargetDevice extends Device >
    {
        void run(ExecutionCall<TargetDevice> call );
    }

    ExecutionOn< TargetDevice > getExecution();

    int arity();

}
