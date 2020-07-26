package neureka.calculus.environment;

import neureka.Tsr;
import neureka.acceleration.Device;

import java.util.List;
import java.util.function.Consumer;

/**
 * This class is the abstract representation of an algorithm storing
 * source code for specific device types and device specific instances
 * of the Execution interface which are responsible for calling
 * the device and executing it!
 * Hens the name: Execution
 *
 * The OpenCLDevice class for example takes the kernel
 * provided by an instance of this class in order to compile it...
 *
 */
public interface OperationTypeImplementation<FinalType>
{
    interface CallPipe
    {
        ExecutionCall<Device> run(ExecutionCall<Device> call);
    }

    <D extends Device, E extends ExecutorFor<D>> FinalType setExecutor(Class<E> deviceClass, E execution);

    <D extends Device, E extends ExecutorFor<D>> ExecutorFor getExecutor(Class<E> deviceClass);

    boolean canHandle(ExecutionCall<Device> call);

    Tsr reduce( ExecutionCall<Device> call, Consumer<ExecutionCall<Device>> finalExecution );

    List< CallPipe > getCallPipeline();
}
