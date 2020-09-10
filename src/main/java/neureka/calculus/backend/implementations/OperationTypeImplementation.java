package neureka.calculus.backend.implementations;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.autograd.ADAgent;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.executions.ExecutorFor;
import neureka.calculus.frontend.AbstractFunction;

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
    String getName();

    interface SuitabilityChecker {
        float canHandle( ExecutionCall call );
    }

    float isImplementationSuitableFor( ExecutionCall call );

    interface DeviceFinder {
        Device findFor( ExecutionCall call );
    }

    Device findDeviceFor( ExecutionCall call );

    interface ForwardADAnalyzer {
        boolean allowsForward( ExecutionCall call );
    }

    boolean canImplementationPerformForwardADFor(ExecutionCall call );

    interface BackwardADAnalyzer {
        boolean allowsBackward( ExecutionCall call );
    }

    boolean canImplementationPerformBackwardADFor(ExecutionCall call );

    interface ADAgentSupplier {
        ADAgent getADAgentOf(
                neureka.calculus.Function f,
                ExecutionCall<Device> call,
                boolean forward
        );
    }

    ADAgent supplyADAgentFor(
            neureka.calculus.Function f,
            ExecutionCall<Device> call,
            boolean forward
    );

    interface InitialCallHook {
        Tsr handle( AbstractFunction caller,  ExecutionCall call );
    }

    Tsr handleInsteadOfDevice(  AbstractFunction caller, ExecutionCall call );

    interface RecursiveJunctionAgent {
        Tsr handle( ExecutionCall call, Function<ExecutionCall, Tsr> goDeeperWith );
    }

    Tsr handleRecursivelyAccordingToArity( ExecutionCall call, Function<ExecutionCall, Tsr> goDeeperWith );

    interface DrainInstantiation {
        ExecutionCall handle( ExecutionCall call );
    }

    ExecutionCall instantiateNewTensorsForExecutionIn( ExecutionCall call );


    Tsr recursiveReductionOf(ExecutionCall<Device> call, Consumer<ExecutionCall<Device>> finalExecution );

    <D extends Device, E extends ExecutorFor<D>> FinalType setExecutor(Class<E> deviceClass, E execution);

    <D extends Device, E extends ExecutorFor<D>> ExecutorFor getExecutor(Class<E> deviceClass);



}
