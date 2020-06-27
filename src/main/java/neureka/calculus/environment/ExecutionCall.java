package neureka.calculus.environment;

import neureka.Tsr;
import neureka.acceleration.Device;

public class ExecutionCall<DeviceType extends Device>
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