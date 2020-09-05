package neureka.calculus.backend.implementations;


import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.backend.*;
import neureka.calculus.backend.executions.ExecutorFor;
import neureka.calculus.frontend.AbstractFunction;

import java.util.HashMap;
import java.util.Map;

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
    private final String _name;

    protected final Map< Class<ExecutorFor< Device >>, ExecutorFor< Device > > _executions = new HashMap<>();

    private SuitabilityChecker _suitabilityChecker;
    private ADAnalyzer _analyzer;
    private ADAgentSupplier _adaCreator;
    private InitialCallHook _hook;
    private RecursiveJunctionAgent _RJAgent;
    private DrainInstantiation _instantiation;

    public AbstractFunctionalOperationTypeImplementation(String name) { _name = name; }

    @Override
    public String getName(){
        return _name;
    }

    //---

    @Override
    public boolean isImplementationSuitableFor(ExecutionCall call) {
        return _suitabilityChecker.canHandle(call);
    }

    public FinalType setSuitabilityChecker(SuitabilityChecker checker) {
        _suitabilityChecker = checker;
        return (FinalType) this;
    }

    //---

    @Override
    public boolean canImplementationPerformADFor(ExecutionCall call) {
        return _analyzer.allowsForward(call);
    }

    public ADAnalyzer getADAnalyzer(){
        return _analyzer;
    }

    public FinalType setADAnalyzer(ADAnalyzer analyzer) {
        _analyzer = analyzer;
        return (FinalType) this;
    }

    //---

    @Override
    public ADAgent supplyADAgentFor(Function f, ExecutionCall<Device> call, boolean forward) {
        return _adaCreator.getADAgentOf(f, call, forward);
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
        return _hook.handle(caller, call);
    }

    public InitialCallHook getCallHook(){
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

    public RecursiveJunctionAgent getRJAgent(){
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

    public DrainInstantiation getDrainInstantiation(){
        return _instantiation;
    }

    public FinalType setDrainInstantiation(DrainInstantiation drainInstantiation) {
        _instantiation = drainInstantiation;
        return (FinalType) this;
    }

    //---

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

    //---

}


