package neureka.calculus.environment;

import neureka.Tsr;
import neureka.acceleration.Device;

import java.util.function.Consumer;
import java.util.function.Function;

/**
 *   This class is the middle layer of the 3 tier abstraction architecture
 *   of Neureka's operation implementations.
 *
 *   Conceptually an implementation of this interface represents "a way of execution" for
 *   the OperationType to which an instance of said implementation would belong.
 *   The "+" operator for example has different OperationTypeImplementation instances
 *   for different ExecutionCall instances.
 *   Tensors within an execution call having the same shape would
 *   trigger the Operation instance of the OperationType, whereas otherwise
 *   the Convolution or Broadcast implementation might be called.
 */
public interface OperationTypeImplementation<FinalType>
{
    interface ADAnalyzer {
        boolean allowsForward(Tsr[] inputs);
    }

    interface RecursiveJunctionAgent {
        Tsr handle( ExecutionCall call, Function<ExecutionCall, Tsr> goDeeperWith );
    }

    ADAnalyzer getADAnalyzer();

    RecursiveJunctionAgent getRJAgent();

    <D extends Device, E extends ExecutorFor<D>> FinalType setExecutor(Class<E> deviceClass, E execution);

    <D extends Device, E extends ExecutorFor<D>> ExecutorFor getExecutor(Class<E> deviceClass);

    boolean canHandle(ExecutionCall<Device> call);

    Tsr recursiveReductionOf(ExecutionCall<Device> call, Consumer<ExecutionCall<Device>> finalExecution );

}
