package neureka.calculus.environment;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.calculus.factory.AbstractFunction;

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
    interface HandleChecker {
        boolean canHandle( ExecutionCall call );
    }

    interface ADAnalyzer {
        boolean allowsForward( ExecutionCall call );
    }

    interface InitialCallHook {
        Tsr handle( AbstractFunction caller,  ExecutionCall call );
    }

    interface RecursiveJunctionAgent {
        Tsr handle( ExecutionCall call, Function<ExecutionCall, Tsr> goDeeperWith );
    }

    interface DrainInstantiation {
        ExecutionCall handle( ExecutionCall call );
    }

    HandleChecker getHandleChecker();
    FinalType setHandleChecker( HandleChecker checker );

    ADAnalyzer getADAnalyzer();
    FinalType setADAnalyzer( ADAnalyzer analyzer );

    InitialCallHook getCallHook();
    FinalType setCallHock( InitialCallHook hook );

    RecursiveJunctionAgent getRJAgent();
    FinalType setRJAgent( RecursiveJunctionAgent rja );

    DrainInstantiation getDrainInstantiation();
    FinalType setDrainInstantiation( DrainInstantiation drainInstantiation );

    <D extends Device, E extends ExecutorFor<D>> FinalType setExecutor(Class<E> deviceClass, E execution);

    <D extends Device, E extends ExecutorFor<D>> ExecutorFor getExecutor(Class<E> deviceClass);

    Tsr recursiveReductionOf(ExecutionCall<Device> call, Consumer<ExecutionCall<Device>> finalExecution );

}
