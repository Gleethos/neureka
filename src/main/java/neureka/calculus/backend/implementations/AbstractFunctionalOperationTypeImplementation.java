package neureka.calculus.backend.implementations;


import neureka.Tsr;
import neureka.devices.Device;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.backend.*;
import neureka.calculus.frontend.AbstractFunction;

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
public abstract class AbstractFunctionalOperationTypeImplementation< FinalType > extends AbstractBaseOperationTypeImplementation< FinalType >
{

    private SuitabilityChecker _suitabilityChecker;
    private DeviceFinder _finder;
    private ForwardADAnalyzer _forwardAnalyzer;
    private BackwardADAnalyzer _backwardAnalyzer;
    private ADAgentSupplier _adaCreator;
    private InitialCallHook _hook;
    private RecursiveJunctionAgent _RJAgent;
    private DrainInstantiation _instantiation;

    public AbstractFunctionalOperationTypeImplementation(String name) {
        super(name);
    }

    //---

    @Override
    public float isImplementationSuitableFor(ExecutionCall call) {
        return _suitabilityChecker.canHandle(call);
    }

    public FinalType setSuitabilityChecker(SuitabilityChecker checker) {
        _suitabilityChecker = checker;
        return (FinalType) this;
    }

    ///---

    @Override
    public Device findDeviceFor(ExecutionCall call) {
        return ( _finder == null ) ? null :_finder.findFor(call);
    }

    public FinalType setDeviceFinder( DeviceFinder finder ) {
        _finder = finder;
        return (FinalType) this;
    }

    //---

    @Override
    public boolean canImplementationPerformForwardADFor(ExecutionCall call ) {
        return _forwardAnalyzer.allowsForward(call);
    }

    public ForwardADAnalyzer getForwardADAnalyzer() {
        return _forwardAnalyzer;
    }

    public FinalType setForwardADAnalyzer(ForwardADAnalyzer analyzer) {
        _forwardAnalyzer = analyzer;
        return (FinalType) this;
    }

    //---

    @Override
    public boolean canImplementationPerformBackwardADFor(ExecutionCall call ) {
        return _backwardAnalyzer.allowsBackward(call);
    }

    public BackwardADAnalyzer getBackwardADAnalyzer() {
        return _backwardAnalyzer;
    }

    public FinalType setBackwardADAnalyzer(BackwardADAnalyzer analyzer) {
        _backwardAnalyzer = analyzer;
        return (FinalType) this;
    }

    //---

    @Override
    public ADAgent supplyADAgentFor(Function f, ExecutionCall<Device> call, boolean forward) {
        return _adaCreator.getADAgentOf( f, call, forward );
    }

    public ADAgentSupplier getADAgentSupplier() {
        return _adaCreator;
    }

    public FinalType setADAgentSupplier(ADAgentSupplier creator ) {
        _adaCreator = creator;
        return (FinalType) this;
    }


    //---

    @Override
    public Tsr handleInsteadOfDevice(AbstractFunction caller, ExecutionCall call) {
        return _hook.handle( caller, call );
    }

    public InitialCallHook getCallHook() {
        return _hook;
    }

    public FinalType setCallHock(InitialCallHook hook) {
        _hook = hook;
        return (FinalType) this;
    }


    //---

    @Override
    public Tsr handleRecursivelyAccordingToArity(ExecutionCall call, java.util.function.Function<ExecutionCall, Tsr> goDeeperWith) {
        return _RJAgent.handle(call, goDeeperWith);
    }

    public RecursiveJunctionAgent getRJAgent() {
        return _RJAgent;
    }

    public FinalType setRJAgent(RecursiveJunctionAgent rja) {
        _RJAgent = rja;
        return (FinalType) this;
    }

    //---

    @Override
    public ExecutionCall instantiateNewTensorsForExecutionIn(ExecutionCall call) {
        return _instantiation.handle(call);
    }

    public DrainInstantiation getDrainInstantiation() {
        return _instantiation;
    }

    public FinalType setDrainInstantiation(DrainInstantiation drainInstantiation) {
        _instantiation = drainInstantiation;
        return (FinalType) this;
    }


    //---

}


