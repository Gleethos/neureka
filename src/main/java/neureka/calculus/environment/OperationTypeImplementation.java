package neureka.calculus.environment;

import neureka.Tsr;
import neureka.acceleration.Device;

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
    class ExecutionCall<DeviceType extends Device>
    {
        private final DeviceType _device;
        private final Tsr[] _tsrs;
        private final int _d;
        private final OperationType _type;

        public ExecutionCall(
                DeviceType device,
                Tsr[] tsrs,
                int d,
                OperationType type
        ) {
            _device = device;
            _tsrs = tsrs;
            _d = d;
            _type = type;
        }
        public DeviceType getDevice() {return _device;}
        public Tsr[] getTensors() {return _tsrs;}
        public Tsr getTensor(int i) {return _tsrs[i];}
        public int getDerivativeIndex() {return _d;}
        public OperationType getType() {return _type;}
        public OperationTypeImplementation getExecutor() { return _type.executorOf(this); }
    }

    <D extends Device, E extends ExecutorFor<D>> FinalType setExecution(Class<E> deviceClass, E execution);

    <D extends Device, E extends ExecutorFor<D>> ExecutorFor getExecution(Class<E> deviceClass);

    boolean canHandle(OperationTypeImplementation.ExecutionCall<Device> call);

    Tsr reduce( OperationTypeImplementation.ExecutionCall<Device> call, Consumer<ExecutionCall<Device>> finalExecution );

}
