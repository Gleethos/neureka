package neureka.calculus.environment.implementations;


import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.autograd.GraphNode;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.ExecutorFor;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.OperationTypeImplementation;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

/**
 * This is the base class for implementations of the OperationTypeImplementation interface.
 * The class implements the component logic required by said interface.
 * Additionally it contains useful methods used to process passed arguments of execution calls.
 *
 * Conceptually this class represents "a way of execution" for
 * the OperationType to which an instance of this class would belong.
 * The "+" operator for example has different OperationTypeImplementation instances
 * for different ExecutionCall instances.
 * Tensors within an execution call having the same shape would
 * trigger the Operation instance of the OperationType, whereas otherwise
 * the Convolution or Broadcast implementation might be called.
 *
 * @param <FinalType> The final type extending this class.
 */
public abstract class AbstractOperationTypeImplementation< FinalType > implements OperationTypeImplementation< FinalType >
{
    private final String _name;

    protected final Map< Class< ExecutorFor< Device > >, ExecutorFor< Device > > _executions = new HashMap<>();

    private SuitabilityChecker _suitabilityChecker;
    private ADAnalyzer _analyzer;
    private ADAgentSupplier _adaCreator;
    private InitialCallHook _hook;
    private RecursiveJunctionAgent _RJAgent;
    private DrainInstantiation _instantiation;

    public AbstractOperationTypeImplementation(String name) { _name = name; }

    @Override
    public String getName(){
        return _name;
    }

    @Override
    public SuitabilityChecker getSuitabilityChecker() {
        return _suitabilityChecker;
    }

    @Override
    public FinalType setSuitabilityChecker(SuitabilityChecker checker) {
        _suitabilityChecker = checker;
        return (FinalType) this;
    }

    @Override
    public ADAnalyzer getADAnalyzer(){
        return _analyzer;
    }

    @Override
    public FinalType setADAnalyzer(ADAnalyzer analyzer) {
        _analyzer = analyzer;
        return (FinalType) this;
    }

    @Override
    public ADAgentSupplier getADAgentCreator() {
        return _adaCreator;
    }

    @Override
    public FinalType setADAgentCreator( ADAgentSupplier creator ) {
        _adaCreator = creator;
        return (FinalType) this;
    }

    @Override
    public InitialCallHook getCallHook(){
        return _hook;
    }

    @Override
    public FinalType setCallHock(InitialCallHook hook) {
        _hook = hook;
        return (FinalType) this;
    }

    @Override
    public RecursiveJunctionAgent getRJAgent(){
        return _RJAgent;
    }

    @Override
    public FinalType setRJAgent(RecursiveJunctionAgent rja) {
        _RJAgent = rja;
        return (FinalType) this;
    }

    @Override
    public DrainInstantiation getDrainInstantiation(){
        return _instantiation;
    }

    @Override
    public FinalType setDrainInstantiation(DrainInstantiation drainInstantiation) {
        _instantiation = drainInstantiation;
        return (FinalType) this;
    }

    @Override
    public <D extends Device, E extends ExecutorFor<D>> FinalType setExecutor(Class<E> deviceClass, E execution){
        _executions.put(
                (Class<ExecutorFor<Device>>) deviceClass,
                (ExecutorFor<Device>) execution
        );
        return (FinalType) this;
    }

    @Override
    public <D extends Device, E extends ExecutorFor<D>> E getExecutor(Class<E> deviceClass){
        return (E) _executions.get(deviceClass);
    }

    @Override
    public Tsr recursiveReductionOf(
            ExecutionCall<Device> call,
            Consumer<ExecutionCall<Device>> finalExecution
    ) {
        Device device = call.getDevice();
        Tsr[] tsrs = call.getTensors();
        int d = call.getDerivativeIndex();
        OperationType type = call.getType();

        Consumer<Tsr>[] rollbacks = new Consumer[tsrs.length];
        for (int i=0; i<tsrs.length; i++) {
            if ( tsrs[i] != null && !tsrs[i].isOutsourced() ) {
                device.add(tsrs[i]);
                rollbacks[i] = device::get;
            } else {
                rollbacks[i] = t -> {};
            }
        }

        /* For the following operations with the correct arity RJAgent should do: ...
            case ("s" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};
            case ("d" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};
            case ("p" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};
            case ("m" + ((char) 187)): tsrs = new Tsr[]{tsrs[2], tsrs[1], tsrs[0]};
            case ">": tsrs = new Tsr[]{tsrs[1], tsrs[0]};
         */
        /*
            Below is the core lambda of recursive preprocessing
            which is defined for each OperationImplementation individually :
         */
        Tsr result = _RJAgent.handle(call, c -> recursiveReductionOf( c, finalExecution ));
        if ( result == null ) {
            finalExecution.accept(
                    new ExecutionCall<>( device, call.getTensors(), d, type )
            );
        } else return result;


        for ( int i = 0; i < tsrs.length; i++ ) {
            if ( tsrs[i] != null && !tsrs[i].isUndefined() ) rollbacks[i].accept(tsrs[i]);
        }
        return tsrs[0];
    }

    public static class Utility
    {
        public static Tsr[] _subset(Tsr[] tsrs, int padding, int index, int offset) {
            if ( offset < 0 ) {
                index += offset;
                offset *= -1;
            }
            Tsr[] newTsrs = new Tsr[offset+padding];
            System.arraycopy(tsrs, index, newTsrs, padding, offset);
            return newTsrs;
        }
        public static Tsr[] _without(Tsr[] tsrs, int index){
            Tsr[] newTsrs = new Tsr[tsrs.length-1];
            for ( int i = 0; i < newTsrs.length; i++ ) newTsrs[i] = tsrs[i+( ( i < index )? 0 : 1 )];
            return newTsrs;
        }

        public static Tsr[] _offsetted(Tsr[] tsrs, int offset){
            Tsr[] newTsrs = new Tsr[tsrs.length-offset];
            newTsrs[0] = Tsr.Create.newTsrLike(tsrs[1]);
            if ( !tsrs[1].has(GraphNode.class ) && tsrs[1] != tsrs[0] ) {//Deleting intermediate results!
                tsrs[1].delete();
                tsrs[1] = null;
            }
            if ( !tsrs[2].has(GraphNode.class) && tsrs[2] != tsrs[0] ) {//Deleting intermediate results!
                tsrs[2].delete();
                tsrs[2] = null;
            }
            System.arraycopy(tsrs, 1+offset, newTsrs, 1, tsrs.length-1-offset);
            newTsrs[1] = tsrs[0];
            return newTsrs;
        }

    }






}


