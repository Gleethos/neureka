package neureka.backend.api.implementations;


import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.implementations.FunctionNode;
import neureka.devices.Device;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;

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
@Accessors( prefix = {"_"}, chain = true )
public abstract class AbstractFunctionalOperationTypeImplementation< FinalType > extends AbstractBaseOperationTypeImplementation< FinalType >
{
    @Getter @Setter private SuitabilityChecker _suitabilityChecker;
    @Getter @Setter private DeviceFinder _deviceFinder;
    @Getter @Setter private ForwardADAnalyzer _forwardADAnalyzer;
    @Getter @Setter private BackwardADAnalyzer _backwardADAnalyzer;
    @Getter @Setter private ADAgentSupplier _ADAgentSupplier;
    @Getter @Setter private InitialCallHook _callHook;
    @Getter @Setter private RecursiveJunctionAgent _RJAgent;
    @Getter @Setter private DrainInstantiation _drainInstantiation;

    public AbstractFunctionalOperationTypeImplementation( String name ) {
        super(name);
    }

    //---

    @Override
    public float isImplementationSuitableFor( ExecutionCall call ) {
        return _suitabilityChecker.canHandle(call);
    }

    ///---

    @Override
    public Device findDeviceFor( ExecutionCall call ) {
        return ( _deviceFinder == null ) ? null : _deviceFinder.findFor(call);
    }

    //---

    @Override
    public boolean canImplementationPerformForwardADFor( ExecutionCall call ) {
        return _forwardADAnalyzer.allowsForward(call);
    }

    //---

    @Override
    public boolean canImplementationPerformBackwardADFor( ExecutionCall call ) {
        return _backwardADAnalyzer.allowsBackward(call);
    }

    //---

    @Override
    public ADAgent supplyADAgentFor( Function f, ExecutionCall<Device> call, boolean forward ) {
        return _ADAgentSupplier.getADAgentOf( f, call, forward );
    }

    //---

    @Override
    public Tsr handleInsteadOfDevice( FunctionNode caller, ExecutionCall call ) {
        return _callHook.handle( caller, call );
    }

    //---

    @Override
    public Tsr handleRecursivelyAccordingToArity( ExecutionCall call, java.util.function.Function<ExecutionCall, Tsr> goDeeperWith ) {
        return _RJAgent.handle(call, goDeeperWith);
    }

    //---

    @Override
    public ExecutionCall instantiateNewTensorsForExecutionIn( ExecutionCall call ) {
        return _drainInstantiation.handle(call);
    }

    //---

    public FinalType build() {
        return (FinalType) this;
    }

}


