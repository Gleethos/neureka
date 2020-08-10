package neureka.calculus.environment;

import neureka.Tsr;
import neureka.acceleration.Device;

/**
 * This class is a simple container holding relevant
 * arguments needed to execute on a targeted Device which
 * is specified by the type parameter below.
 *
 * @param <DeviceType> The Device implementation targeted by an instance of this ExecutionCall!
 */
public class ExecutionCall<DeviceType extends Device>
{
    private final DeviceType _device;
    private final Tsr[] _tsrs;
    private final int _d;
    private final OperationType _type;
    private OperationTypeImplementation _implementation;

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
        _implementation = null;
    }
    public DeviceType getDevice() {return _device;}
    public Tsr[] getTensors() {return _tsrs;}
    public Tsr getTensor(int i) {return _tsrs[i];}
    public int getDerivativeIndex() {return _d;}
    public OperationType getType() {return _type;}
    public OperationTypeImplementation getImplementation() {
        if ( _implementation != null ) return _implementation;
        else _implementation = _type.implementationOf(this);
        return _implementation;
    }
}