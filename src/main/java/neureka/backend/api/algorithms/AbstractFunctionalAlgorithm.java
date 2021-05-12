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
public abstract class AbstractFunctionalAlgorithm< FinalType extends Algorithm<FinalType> > extends AbstractBaseAlgorithm< FinalType >
{
    @Getter @Setter private Algorithm.SuitabilityChecker _isSuitableFor;
    @Getter @Setter private Algorithm.DeviceFinder _findDeviceFor;
    @Getter @Setter private Algorithm.ForwardADAnalyzer _canPerformForwardADFor;
    @Getter @Setter private Algorithm.BackwardADAnalyzer _canPerformBackwardADFor;
    @Getter @Setter private Algorithm.ADAgentSupplier _supplyADAgentFor;
    @Getter @Setter private Algorithm.InitialCallHook _handleInsteadOfDevice;
    @Getter @Setter private RecursiveJunctor _handleRecursivelyAccordingToArity;
    @Getter @Setter private Algorithm.DrainInstantiation _instantiateNewTensorsForExecutionIn;

    public AbstractFunctionalAlgorithm(String name ) {
        super(name);
    }

    //---

    @Override
    public float isSuitableFor( ExecutionCall<? extends Device<?>> call ) {
        return _isSuitableFor.canHandle(call);
    }

    ///---

    @Override
    public Device findDeviceFor( ExecutionCall<? extends Device<?>> call ) {
        return ( _findDeviceFor == null ) ? null : _findDeviceFor.findFor(call);
    }

    //---

    @Override
    public boolean canPerformForwardADFor( ExecutionCall<? extends Device<?>> call ) {
        return _canPerformForwardADFor.allowsForward(call);
    }

    //---

    @Override
    public boolean canPerformBackwardADFor( ExecutionCall<? extends Device<?>> call ) {
        return _canPerformBackwardADFor.allowsBackward( call );
    }

    //---

    @Override
    public ADAgent supplyADAgentFor( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) {
        return _supplyADAgentFor.getADAgentOf( f, call, forward );
    }

    //---

    @Override
    public Tsr handleInsteadOfDevice( FunctionNode caller, ExecutionCall<? extends Device<?>> call ) {
        return _handleInsteadOfDevice.handle( caller, call );
    }

    //---

    @Override
    public Tsr<?> handleRecursivelyAccordingToArity( ExecutionCall<? extends Device<?>> call, java.util.function.Function<ExecutionCall<? extends Device<?>>, Tsr<?>> goDeeperWith ) {
        return _handleRecursivelyAccordingToArity.handle( call, goDeeperWith );
    }

    //---

    @Override
    public ExecutionCall instantiateNewTensorsForExecutionIn( ExecutionCall<? extends Device<?>> call ) {
        return _instantiateNewTensorsForExecutionIn.handle( call );
    }

    //---

    public FinalType build() {
        return (FinalType) this;
    }

}


