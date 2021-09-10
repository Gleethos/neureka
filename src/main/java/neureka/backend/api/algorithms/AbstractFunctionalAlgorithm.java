package neureka.backend.api.algorithms;


import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.algorithms.fun.*;
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
 * @param <C> The final type extending this class.
 */
public abstract class AbstractFunctionalAlgorithm< C extends Algorithm<C> > extends AbstractBaseAlgorithm<C>
{
    private SuitabilityChecker _isSuitableFor;
    private ForwardADChecker _canPerformForwardADFor;
    private BackwardADChecker _canPerformBackwardADFor;
    private ADAgentSupplier _supplyADAgentFor;
    private ExecutionOrchestration _handleInsteadOfDevice;
    private ExecutionPreparation _instantiateNewTensorsForExecutionIn;

    public AbstractFunctionalAlgorithm( String name ) {
        super(name);
    }


    //---

    @Override
    public float isSuitableFor( ExecutionCall<? extends Device<?>> call ) {
        return _isSuitableFor.isSuitableFor(call);
    }

    //---

    @Override
    public boolean canPerformForwardADFor( ExecutionCall<? extends Device<?>> call ) {
        return _canPerformForwardADFor.canPerformForwardADFor(call);
    }

    //---

    @Override
    public boolean canPerformBackwardADFor( ExecutionCall<? extends Device<?>> call ) {
        return _canPerformBackwardADFor.canPerformBackwardADFor( call );
    }

    //---

    @Override
    public ADAgent supplyADAgentFor( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) {
        return _supplyADAgentFor.supplyADAgentFor( f, call, forward );
    }

    //---

    @Override
    public Tsr<?> execute(FunctionNode caller, ExecutionCall<? extends Device<?>> call ) {
        return _handleInsteadOfDevice.execute( caller, call );
    }

    //---

    @Override
    public ExecutionCall<? extends Device<?>> prepare( ExecutionCall<? extends Device<?>> call ) {
        return _instantiateNewTensorsForExecutionIn.prepare( call );
    }

    //---

    public C build() { return (C) this; }

    /**
     *  The {@link SuitabilityChecker}
     *  checks if a given instance of an {@link ExecutionCall} is
     *  suitable to be executed in {@link neureka.backend.api.ImplementationFor} instances
     *  residing in this {@link Algorithm} as components.
     *
     * @return This very instance to enable method chaining.
     */
    public AbstractFunctionalAlgorithm<C> setIsSuitableFor( SuitabilityChecker isSuitableFor ) {
        this._isSuitableFor = isSuitableFor;
        return this;
    }

    /**
     *  A {@link ForwardADChecker} lambda checks if this
     *  {@link Algorithm} can perform forward AD for a given {@link ExecutionCall}.
     *
     * @param canPerformForwardADFor
     * @return This very instance to enable method chaining.
     */
    public AbstractFunctionalAlgorithm<C> setCanPerformForwardADFor( ForwardADChecker canPerformForwardADFor ) {
        this._canPerformForwardADFor = canPerformForwardADFor;
        return this;
    }

    /**
     *  A {@link BackwardADChecker} lambda checks if this
     *  {@link Algorithm} can perform backward AD for a given {@link ExecutionCall}.
     *
     * @param canPerformBackwardADFor
     * @return This very instance to enable method chaining.
     */
    public AbstractFunctionalAlgorithm<C> setCanPerformBackwardADFor( BackwardADChecker canPerformBackwardADFor ) {
        this._canPerformBackwardADFor = canPerformBackwardADFor;
        return this;
    }

    /**
     *  This method receives a {@link neureka.backend.api.algorithms.fun.ADAgentSupplier} which will supply
     *  {@link ADAgent} instances which can perform backward and forward auto differentiation.
     *
     * @param supplyADAgentFor
     * @return This very instance to enable method chaining.
     */
    public AbstractFunctionalAlgorithm<C> setSupplyADAgentFor( ADAgentSupplier supplyADAgentFor ) {
        this._supplyADAgentFor = supplyADAgentFor;
        return this;
    }

    public AbstractFunctionalAlgorithm<C> setOrchestration(ExecutionOrchestration handleInsteadOfDevice ) {
        this._handleInsteadOfDevice = handleInsteadOfDevice;
        return this;
    }

    public AbstractFunctionalAlgorithm<C> setInstantiateNewTensorsForExecutionIn( ExecutionPreparation instantiateNewTensorsForExecutionIn ) {
        this._instantiateNewTensorsForExecutionIn = instantiateNewTensorsForExecutionIn;
        return this;
    }
}


