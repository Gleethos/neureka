package neureka.calculus.backend;

import neureka.Tsr;
import neureka.device.Device;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.backend.implementations.OperationTypeImplementation;
import neureka.calculus.backend.operations.OperationType;

import java.util.Map;
import java.util.TreeMap;

/**
 * This class is a simple container holding relevant
 * arguments needed to execute on a targeted Device which
 * is specified by the type parameter below.
 *
 * It also holds a context map responsible for storing
 * operation specific variables.
 *
 * @param <DeviceType> The Device implementation targeted by an instance of this ExecutionCall!
 */
public class ExecutionCall< DeviceType extends Device >
{
    public interface Mutator {
        Tsr[] mutate( Tsr[] tensors );
    }

    private final DeviceType _device;
    private Tsr[] _tensors;
    private final int _d;
    private int _j = -1;
    private final OperationType _type;
    private OperationTypeImplementation<OperationTypeImplementation> _implementation;

    private Map<String, Object> _context;

    public ExecutionCall(
            DeviceType device,
            Tsr[] tensors,
            int d,
            OperationType type
    ) {
        _device = device;
        _tensors = tensors;
        _d = d;
        _type = type;
        _implementation = null;
        _context = null;
    }
    
    public ExecutionCall(
            DeviceType device,
            Tsr[] tensors,
            int d,
            int j,
            OperationType type
    ) {
        _device = device;
        _tensors = tensors;
        _d = d;
        _j = j;
        _type = type;
        _implementation = null;
    }
    
    public int getJ() {
        return _j;
    }
    
    public DeviceType getDevice() {return _device;}
    
    public Tsr[] getTensors() {return _tensors;}
    
    public Tsr getTensor(int i) {return _tensors[i];}

    /**
     * This method returns an import property whose
     * role might not be clear at first :
     * An operation can have multiple inputs, however
     * when calculating the derivative for a forward or backward pass
     * then one must know which derivative ought to be calculated.
     * So the "derivative index" targets said input.
     * This property is -1 when no derivative should be calculated,
     * however 0... when targeting an input to calculate the derivative of.
     *
     * @return The index of the input whose derivative ought to be calculated.
     */
    public int getDerivativeIndex() {return _d;}
    
    public OperationType getType() {return _type;}
    
    public OperationTypeImplementation getImplementation() {
        if ( _implementation != null ) return _implementation;
        else _implementation = _type.implementationOf(this);
        return _implementation;
    }
    
    public boolean allowsForward(){
        return getImplementation().canImplementationPerformForwardADFor(this);
    }

    public boolean allowsBackward(){
        return getImplementation().canImplementationPerformBackwardADFor(this);
    }

    public ADAgent getADAgentFrom(Function function, Tsr derivative, ExecutionCall<Device> call, boolean forward ) {
        if ( this._context != null ) {
            if ( call._context ==null ) call._context = new TreeMap<>();
            call._context.putAll(this._context);
        }
        if( derivative != null ) assert (call._context != null && call._context.containsKey("derivative"));
        else assert call._context == null || !call._context.containsKey("derivative");
        return getImplementation().supplyADAgentFor(function, call, forward);
    }
    
    public void mutateArguments(Mutator mutation){
        _tensors = mutation.mutate(_tensors);
    }
    
    public ExecutionCall<DeviceType> withNew(Tsr[] tensors) {
        return new ExecutionCall<DeviceType>(_device, tensors, _d, _j, _type);
    }

    public ExecutionCall<DeviceType> withNew(DeviceType device) {
        return new ExecutionCall<DeviceType>(device, _tensors, _d, _j, _type);
    }

    public <T> T getAt(Class<T> type){
        if ( _context == null ) return null;
        return (T) _context.get(getClass().getName());
    }

    public Object getAt(String varName){
        if ( _context == null ) return null;
        return _context.get(varName);
    }

    public <T> ExecutionCall<DeviceType> putAt(String s, T o){
        if ( _context == null ) _context = new TreeMap<>();
        _context.put(s,o);
        return this;
    }

    public Map<String, Object> getContext(){
        return _context;
    }

    public void takeContext( Map<String, Object>  context ){
        if(_context==null && context!=null )_context = new TreeMap<>();
        if(context!=null) _context.putAll(_context);
    }


}