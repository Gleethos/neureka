package neureka.calculus.environment.executors;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.calculus.environment.OperationType;

import java.util.function.Consumer;

/**
 * TODO:
 *
 * This class is the representation a kernel implementation
 * of a device!
 * Instances of this class provide the source for
 * for devices to use!
 * The OpenCLDevice class for example takes the kernel
 * provided by an instance of this class in order to compile it...
 *
 * Calls to the device are implemented via implementations of Execution interface!
 *
 */

public interface TypeExecutor
{
    class ExecutionCall
    {
        private final Device _device;
        private final Tsr[] _tsrs;
        private final int _d;
        private final OperationType _type;
        private final TypeExecutor _executor;

        public ExecutionCall(Device device, Tsr[] tsrs, int d, OperationType type)
        {
            _device = device;
            _tsrs = tsrs;
            _d = d;
            _type = type;
            _executor = _type.executorOf(this);
        }
        public Device getDevice() {return _device;}
        public Tsr[] getTensors() {return _tsrs;}
        public int getDerivativeIndex() {return _d;}
        public OperationType getType() {return _type;}
        public TypeExecutor getExecutor() { return _executor; }
    }

    <T extends Execution> TypeExecutor setExecution(Device device, T execution);

    <T extends Execution> T getExecution(Device device);

    boolean canHandle(TypeExecutor.ExecutionCall call);

    public Tsr reduce(TypeExecutor.ExecutionCall call, Consumer<ExecutionCall> finalExecution);


}
