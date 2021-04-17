package neureka.backend.api.algorithms;


import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;
import neureka.Tsr;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.calculus.implementations.FunctionNode;
import neureka.devices.Device;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;

/**
 *  This is the base class for implementations of the {@link Algorithm} interface.
 *  The class implements a basic component system, as is implicitly expected by said interface.
 *  Additionally it contains useful methods used to process passed arguments of {@link ExecutionCall}
 *  as well as an implementation of the {@link Algorithm} interface which allows its methods to
 *  be implemented in a functional programming style, meaning that instances of concrete implementations
 *  extending this abstract class have setters for lambdas representing the {@link Algorithm} methods.
 *  It is being used by the standard backend of Neureka as abstract base class for various algorithms.
 *
 *  Conceptually an implementation of the {@link Algorithm} interface represents "a sub-kind of operation" for
 *  an instance of an implementation of the {@link Operation} interface. <br>
 *  The "+" operator for example has different {@link Algorithm} instances tailored to specific requirements
 *  originating from different {@link ExecutionCall} instances with unique arguments.
 *  {@link Tsr} instances within an execution call having the same shape would
 *  cause the {@link Operation} instance to chose an {@link Algorithm} instance which is responsible
 *  for performing elementwise operations, whereas otherwise the {@link neureka.backend.standard.algorithms.Broadcast}
 *  algorithm might be called to perform the operation.
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


