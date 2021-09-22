package neureka.backend.api.algorithms;


import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.algorithms.fun.*;
import neureka.calculus.Function;
import neureka.calculus.RecursiveExecutor;
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
    private SuitabilityPredicate _isSuitableFor;
    private ForwardADPredicate _canPerformForwardADFor;
    private BackwardADPredicate _canPerformBackwardADFor;
    private ADAgentSupplier _supplyADAgentFor;
    private ExecutionDispatcher _handleInsteadOfDevice;
    private ExecutionPreparation _instantiateNewTensorsForExecutionIn;

    public AbstractFunctionalAlgorithm( String name ) { super(name); }

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
    public ADAgent supplyADAgentFor(Function function, ExecutionCall<? extends Device<?>> call, boolean forward ) {
        return _supplyADAgentFor.supplyADAgentFor(function, call, forward );
    }

    //---

    /**
     *  This method receives an {@link ExecutionDispatcher} lambda which
     *  is the final execution procedure responsible for electing an {@link neureka.backend.api.ImplementationFor}
     *  the chosen {@link Device} in a given {@link ExecutionCall}.
     *  However the  {@link ExecutionDispatcher} does not have to select a device specific implementation.
     *  It can also occupy the rest of the execution without any other steps being taken.
     *  For example, a {@link neureka.backend.api.ImplementationFor} or a {@link RecursiveExecutor}
     *  would not be used if not explicitly called.
     *  Bypassing other procedures is useful for full control and of course to implement unorthodox types of operations
     *  like the {@link neureka.backend.standard.operations.other.Reshape} operation
     *  which is very different from classical operations.
     *  Although the {@link ExecutionCall} passed to implementations of this will contain
     *  a fairly suitable {@link Device} assigned to a given {@link neureka.backend.api.Algorithm},
     *  one can simply ignore it and find a custom one which fits the contents of the given
     *  {@link ExecutionCall} instance better.
     *
     * @param caller The {@link FunctionNode} from which this request for execution emerged.
     * @param call The {@link ExecutionCall} whose contents ought to be executed.
     * @return The result of the execution.
     */
    @Override
    public Tsr<?> dispatch( FunctionNode caller, ExecutionCall<? extends Device<?>> call ) {
        return _handleInsteadOfDevice.dispatch( caller, call );
    }

    //---

    @Override
    public ExecutionCall<? extends Device<?>> prepare( ExecutionCall<? extends Device<?>> call ) {
        return _instantiateNewTensorsForExecutionIn.prepare( call );
    }

    //---

    public C build() { return (C) this; }

    /**
     *  The {@link SuitabilityPredicate}
     *  checks if a given instance of an {@link ExecutionCall} is
     *  suitable to be executed in {@link neureka.backend.api.ImplementationFor} instances
     *  residing in this {@link Algorithm} as components.
     *
     * @return This very instance to enable method chaining.
     */
    public AbstractFunctionalAlgorithm<C> setIsSuitableFor( SuitabilityPredicate isSuitableFor ) {
        this._isSuitableFor = isSuitableFor;
        return this;
    }

    /**
     *  A {@link ForwardADPredicate} lambda checks if this
     *  {@link Algorithm} can perform forward AD for a given {@link ExecutionCall}.
     *
     * @param canPerformForwardADFor
     * @return This very instance to enable method chaining.
     */
    public AbstractFunctionalAlgorithm<C> setCanPerformForwardADFor( ForwardADPredicate canPerformForwardADFor ) {
        this._canPerformForwardADFor = canPerformForwardADFor;
        return this;
    }

    /**
     *  A {@link BackwardADPredicate} lambda checks if this
     *  {@link Algorithm} can perform backward AD for a given {@link ExecutionCall}.
     *
     * @param canPerformBackwardADFor
     * @return This very instance to enable method chaining.
     */
    public AbstractFunctionalAlgorithm<C> setCanPerformBackwardADFor( BackwardADPredicate canPerformBackwardADFor ) {
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

    public AbstractFunctionalAlgorithm<C> setExecutionDispatcher(ExecutionDispatcher handleInsteadOfDevice ) {
        this._handleInsteadOfDevice = handleInsteadOfDevice;
        return this;
    }

    public AbstractFunctionalAlgorithm<C> setCallPreparation( ExecutionPreparation instantiateNewTensorsForExecutionIn ) {
        this._instantiateNewTensorsForExecutionIn = instantiateNewTensorsForExecutionIn;
        return this;
    }
}


