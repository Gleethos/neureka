package neureka.backend.api.algorithms;


import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.calculus.Function;
import neureka.calculus.implementations.FunctionNode;
import neureka.devices.Device;

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
public abstract class AbstractFunctionalAlgorithm< FinalType extends Algorithm<FinalType> > extends AbstractBaseAlgorithm< FinalType >
{
    private Algorithm.SuitabilityChecker _isSuitableFor;
    private Algorithm.DeviceFinder _findDeviceFor;
    private Algorithm.ForwardADAnalyzer _canPerformForwardADFor;
    private Algorithm.BackwardADAnalyzer _canPerformBackwardADFor;
    private Algorithm.ADAgentSupplier _supplyADAgentFor;
    private Algorithm.InitialCallHook _handleInsteadOfDevice;
    private RecursiveJunctor _handleRecursivelyAccordingToArity;
    private Algorithm.DrainInstantiation _instantiateNewTensorsForExecutionIn;

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

    /**
     *  The {@link neureka.backend.api.Algorithm.SuitabilityChecker}
     *  checks if a given instance of an {@link ExecutionCall} is
     *  suitable to be executed in {@link neureka.backend.api.ImplementationFor}
     *  residing in this {@link Algorithm} as components.
     *
     * @return A lambda which checks if a given {@link ExecutionCall} instance is suitable to be executed by this {@link Algorithm}.
     */
    public SuitabilityChecker getIsSuitableFor() {
        return this._isSuitableFor;
    }

    /**
     *  The {@link neureka.backend.api.Algorithm.DeviceFinder} finds
     *  a {@link Device} instance which fits the contents of a given {@link ExecutionCall} instance.
     *  The finder is supposed to find a {@link Device} which can be most easily shared
     *  by the {@link Tsr} instances within the {@link ExecutionCall} that is being received by the finder.
     *
     * @return A finder lambda for finding a suitable {@link Device} implementation instance for a given {@link ExecutionCall} passed to the finder.
     */
    public DeviceFinder getFindDeviceFor() {
        return this._findDeviceFor;
    }

    /**
     *  A {@link neureka.backend.api.Algorithm.ForwardADAnalyzer} lambda checks if this
     *  {@link Algorithm} can perform forward AD for a given {@link ExecutionCall}.
     *
     * @return An analyzer which return a truth value determining if this {@link Algorithm} can perform forward AD on a given {@link ExecutionCall}
     */
    public ForwardADAnalyzer getCanPerformForwardADFor() {
        return this._canPerformForwardADFor;
    }

    /**
     *  A {@link neureka.backend.api.Algorithm.BackwardADAnalyzer} lambda checks if this
     *  {@link Algorithm} can perform backward AD for a given {@link ExecutionCall}.
     *
     * @return An analyzer which return a truth value determining if this {@link Algorithm} can perform backward AD on a given {@link ExecutionCall}
     */
    public BackwardADAnalyzer getCanPerformBackwardADFor() {
        return this._canPerformBackwardADFor;
    }

    /**
     *  This {@link neureka.backend.api.Algorithm.ADAgentSupplier} will supply
     *  {@link ADAgent} instances which can perform backward and forward auto differentiation.
     *
     * @return An {@link neureka.backend.api.Algorithm.ADAgentSupplier} for creting suitable {@link ADAgent} instances.
     */
    public ADAgentSupplier getSupplyADAgentFor() {
        return this._supplyADAgentFor;
    }

    /**
     *  The {@link neureka.backend.api.Algorithm.InitialCallHook} lambda
     *  is simply a bypass procedure which if provided will simply occupy
     *  the rest of the execution without any other steps being taken.
     *  For example the {@link neureka.backend.api.Algorithm.RecursiveJunctor}
     *  would not be used in that case.
     *  This bypassing is useful for unorthodox types of operations
     *  like the {@link neureka.backend.standard.operations.other.Reshape} opertion.
     *
     * @return A lambda which bypasses the default execution procedure.
     */
    public InitialCallHook getHandleInsteadOfDevice() {
        return this._handleInsteadOfDevice;
    }

    /**
     *  This returns an instance of the {@link neureka.backend.api.Algorithm.RecursiveJunctor}
     *  which enables the pairwise execution of a chain of an abitrary number of
     *  {@link ExecutionCall} arguments.
     *  This is useful for simple operators like '+' or '*' which
     *  might have any number of arguments...
     *
     * @return A lambda which will be executed recursively to executes multiple arguments pairwise.
     */
    public RecursiveJunctor getHandleRecursivelyAccordingToArity() {
        return this._handleRecursivelyAccordingToArity;
    }

    /**
     *  An {@link Algorithm} will typically produce a result when executing an {@link ExecutionCall}.
     *  This result must be created somehow.
     *  The {@link neureka.backend.api.Algorithm.DrainInstantiation} lambda instance
     *  returned by this method will do just that...
     *
     * @return A result instantiation lambda called before execution...
     */
    public DrainInstantiation getInstantiateNewTensorsForExecutionIn() {
        return this._instantiateNewTensorsForExecutionIn;
    }

    /**
     *  The {@link neureka.backend.api.Algorithm.SuitabilityChecker}
     *  checks if a given instance of an {@link ExecutionCall} is
     *  suitable to be executed in {@link neureka.backend.api.ImplementationFor} instances
     *  residing in this {@link Algorithm} as components.
     *
     * @return This very instance to enable method chaining.
     */
    public AbstractFunctionalAlgorithm<FinalType> setIsSuitableFor(SuitabilityChecker _isSuitableFor) {
        this._isSuitableFor = _isSuitableFor;
        return this;
    }

    /**
     *  The {@link neureka.backend.api.Algorithm.DeviceFinder} finds
     *  a {@link Device} instance which fits the contents of a given {@link ExecutionCall} instance.
     *  The finder is supposed to find a {@link Device} which can be most easily shared
     *  by the {@link Tsr} instances within the {@link ExecutionCall} that is being received by the finder.
     *
     * @param _findDeviceFor The finder lambda which ought to find a suitable device for a given {@link ExecutionCall}.
     * @return This very instance to enable method chaining.
     */
    public AbstractFunctionalAlgorithm<FinalType> setFindDeviceFor(DeviceFinder _findDeviceFor) {
        this._findDeviceFor = _findDeviceFor;
        return this;
    }

    /**
     *  A {@link neureka.backend.api.Algorithm.ForwardADAnalyzer} lambda checks if this
     *  {@link Algorithm} can perform forward AD for a given {@link ExecutionCall}.
     *
     * @param _canPerformForwardADFor
     * @return This very instance to enable method chaining.
     */
    public AbstractFunctionalAlgorithm<FinalType> setCanPerformForwardADFor(ForwardADAnalyzer _canPerformForwardADFor) {
        this._canPerformForwardADFor = _canPerformForwardADFor;
        return this;
    }

    /**
     *  A {@link neureka.backend.api.Algorithm.BackwardADAnalyzer} lambda checks if this
     *  {@link Algorithm} can perform backward AD for a given {@link ExecutionCall}.
     *
     * @param _canPerformBackwardADFor
     * @return This very instance to enable method chaining.
     */
    public AbstractFunctionalAlgorithm<FinalType> setCanPerformBackwardADFor(BackwardADAnalyzer _canPerformBackwardADFor) {
        this._canPerformBackwardADFor = _canPerformBackwardADFor;
        return this;
    }

    /**
     *  This method receives a {@link neureka.backend.api.Algorithm.ADAgentSupplier} which will supply
     *  {@link ADAgent} instances which can perform backward and forward auto differentiation.
     *
     * @param _supplyADAgentFor
     * @return This very instance to enable method chaining.
     */
    public AbstractFunctionalAlgorithm<FinalType> setSupplyADAgentFor(ADAgentSupplier _supplyADAgentFor) {
        this._supplyADAgentFor = _supplyADAgentFor;
        return this;
    }

    public AbstractFunctionalAlgorithm<FinalType> setHandleInsteadOfDevice(InitialCallHook _handleInsteadOfDevice) {
        this._handleInsteadOfDevice = _handleInsteadOfDevice;
        return this;
    }

    public AbstractFunctionalAlgorithm<FinalType> setHandleRecursivelyAccordingToArity(RecursiveJunctor _handleRecursivelyAccordingToArity) {
        this._handleRecursivelyAccordingToArity = _handleRecursivelyAccordingToArity;
        return this;
    }

    public AbstractFunctionalAlgorithm<FinalType> setInstantiateNewTensorsForExecutionIn(DrainInstantiation _instantiateNewTensorsForExecutionIn) {
        this._instantiateNewTensorsForExecutionIn = _instantiateNewTensorsForExecutionIn;
        return this;
    }
}


