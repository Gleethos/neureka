package neureka.backend.api.algorithms;


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
 * This is the base class for implementations of the {@link Algorithm} interface.
 * The class implements the component logic required by said interface.
 * Additionally it contains useful methods used to process passed arguments of {@link ExecutionCall}.
 *
 * Conceptually this class represents "a way of execution" for
 * the Operation to which an instance of this class would belong.
 * The "+" operator for example has different {@link Algorithm} instances
 * for {@link ExecutionCall} instances hosting tensors which might require
 * either elementwise addition or broadcasting depending on the shapes.
 * Tensors within an execution call having the same shape would
 * trigger an elementwise {@link Algorithm} instance stored in the
 * {@link neureka.backend.api.operations.Operation}, whereas otherwise
 * the {@link neureka.backend.standard.algorithms.Convolution}
 * or
 * {@link neureka.backend.standard.algorithms.Broadcast}
 * algorithm type might be called.
 *
 * @param <FinalType> The final type extending this class.
 */
@Accessors( prefix = {"_"}, chain = true )
public abstract class AbstractFunctionalAlgorithm< FinalType > extends AbstractBaseAlgorithm< FinalType >
{
    @Getter @Setter private SuitabilityChecker _suitabilityChecker;
    @Getter @Setter private DeviceFinder _deviceFinder;
    @Getter @Setter private ForwardADAnalyzer _forwardADAnalyzer;
    @Getter @Setter private BackwardADAnalyzer _backwardADAnalyzer;
    @Getter @Setter private ADAgentSupplier _ADAgentSupplier;
    @Getter @Setter private InitialCallHook _callHook;
    @Getter @Setter private RecursiveJunctionAgent _RJAgent;
    @Getter @Setter private DrainInstantiation _drainInstantiation;

    public AbstractFunctionalAlgorithm(String name ) {
        super(name);
    }

    //---

    @Override
    public float isAlgorithmSuitableFor(ExecutionCall call ) {
        return _suitabilityChecker.canHandle(call);
    }

    ///---

    @Override
    public Device findDeviceFor( ExecutionCall call ) {
        return ( _deviceFinder == null ) ? null : _deviceFinder.findFor(call);
    }

    //---

    @Override
    public boolean canAlgorithmPerformForwardADFor(ExecutionCall call ) {
        return _forwardADAnalyzer.allowsForward(call);
    }

    //---

    @Override
    public boolean canAlgorithmPerformBackwardADFor( ExecutionCall call ) {
        return _backwardADAnalyzer.allowsBackward( call );
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
    public Tsr handleRecursivelyAccordingToArity( ExecutionCall call, java.util.function.Function<ExecutionCall, Tsr<?>> goDeeperWith ) {
        return _RJAgent.handle( call, goDeeperWith );
    }

    //---

    @Override
    public ExecutionCall instantiateNewTensorsForExecutionIn( ExecutionCall call ) {
        return _drainInstantiation.handle( call );
    }

    //---

    public FinalType build() {
        return (FinalType) this;
    }

}


