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
        private final OperationTypeImplementation _executor;

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
            _executor = _type.executorOf(this);
        }
        public DeviceType getDevice() {return _device;}
        public Tsr[] getTensors() {return _tsrs;}
        public Tsr getTensor(int i) {return _tsrs[i];}
        public int getDerivativeIndex() {return _d;}
        public OperationType getType() {return _type;}
        public OperationTypeImplementation getExecutor() { return _executor; }
    }

    <T> FinalType setExecution(Class<T> deviceClass, Execution execution);

    <T> Execution getExecution(Class<T> deviceClass);

    boolean canHandle(OperationTypeImplementation.ExecutionCall call);

    Tsr reduce( OperationTypeImplementation.ExecutionCall call, Consumer<ExecutionCall> finalExecution );

    // Call implementation :

    <T extends Device> void callImplementationFor( ExecutionCall<T> call );


}
