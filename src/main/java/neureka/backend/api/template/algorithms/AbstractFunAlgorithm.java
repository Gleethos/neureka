package neureka.backend.api.template.algorithms;

import neureka.backend.api.*;
import neureka.backend.api.fun.ADSupportPredicate;
import neureka.backend.api.fun.Execution;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.main.memory.MemValidator;
import neureka.calculus.Function;
import neureka.devices.Device;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AbstractFunAlgorithm extends AbstractAlgorithm
{
    private static final Logger _LOG = LoggerFactory.getLogger( AbstractFunDeviceAlgorithm.class );
    /*
        Consider the following lambdas as effectively immutable because this
        class will warn us if any field variable is set for a second time.
        This makes the backend somewhat hackable, but also manageable with respect to complexity.
     */
    private SuitabilityPredicate _isSuitableFor;
    private ADSupportPredicate _autogradModeFor;
    private Execution _execution;
    /*
        This flag will ensure that we can warn the user that the state has been illegally modified.
     */
    private boolean _isFullyBuilt = false;


    protected AbstractFunAlgorithm(String name) {
        super(name);
    }

    /**
     *  The {@link SuitabilityPredicate} checks if a given instance of an {@link ExecutionCall} is
     *  suitable to be executed in {@link neureka.backend.api.ImplementationFor}
     *  residing in this {@link Algorithm} as components.
     *  It can be implemented as s simple lambda.
     */
    @Override
    public final float isSuitableFor( ExecutionCall<? extends Device<?>> call ) {
        _checkReadiness();
        return _isSuitableFor.isSuitableFor(call);
    }

    /**
     * @return A new concrete implementation of the {@link AbstractFunDeviceAlgorithm} which
     *         is fully built and ready to be used as an {@link Operation} component.
     */
    public final AbstractFunAlgorithm buildFunAlgorithm() {
        if (
                _isSuitableFor == null ||
                _autogradModeFor == null ||
                _execution == null
        ) {
            throw new IllegalStateException(
                    "Instance '"+getClass().getSimpleName()+"' incomplete!"
            );
        }

        _isFullyBuilt = true;
        return this;
    }

    /**
     *  This method ensures that this algorithm was fully supplied with all the
     *  required lambdas...
     */
    private void _checkReadiness() {
        if ( !_isFullyBuilt ) {
            throw new IllegalStateException(
                    "Trying use an instance of '"+this.getClass().getSimpleName()+"' with name '" + getName() + "' " +
                            "which was not fully built!"
            );
        }
    }

    /**
     *  Neureka is supposed to be extremely modular and in a sense its backend should be "hackable" to a degree.
     *  However, this comes with a lot of risk, because it requires us to expose mutable state, which is not good.
     *  This class is semi-immutable, by simply warning us about any mutations after building was completed!
     *
     * @param newState The state which will be set.
     * @param <T> The type of the thing which is supposed to be set.
     * @param current The state which is currently set.
     * @return The checked thing.
     */
    private <T> T _checked( T newState, T current, Class<T> type ) {
        if ( _isFullyBuilt )
            _LOG.warn(
                "Implementation '" + type.getSimpleName() + "' in algorithm '"+this+"' was modified! " +
                "Please consider only modifying the standard backend state of Neureka for experimental reasons."
            );
        else if ( current != null && newState == null )
            throw new IllegalArgumentException(
                    "Trying set an already specified implementation of lambda '"+current.getClass().getSimpleName()+"' to null!"
            );

        return newState;
    }

    /**
     *  The {@link SuitabilityPredicate} received by this method
     *  checks if a given instance of an {@link ExecutionCall} is
     *  suitable to be executed in {@link neureka.backend.api.ImplementationFor} instances
     *  residing in this {@link Algorithm} as components.
     *  The lambda will be called by the {@link #isSuitableFor(ExecutionCall)} method
     *  by any given {@link Operation} instances this algorithm belongs to.
     *
     * @param isSuitableFor The suitability predicate which determines if the algorithm is suitable or not.
     * @return This very instance to enable method chaining.
     */
    public final AbstractFunAlgorithm setIsSuitableFor(SuitabilityPredicate isSuitableFor ) {
        _isSuitableFor = _checked(isSuitableFor, _isSuitableFor, SuitabilityPredicate.class);
        return this;
    }

    /**
     *  A {@link ADSupportPredicate} lambda checks what kind of auto differentiation mode an
     *  {@link Algorithm} supports for a given {@link ExecutionCall}.
     *  The lambda will be called by the {@link #autoDiffModeFrom(ExecutionCall)} method
     *  by any given {@link Operation} instances this algorithm belongs to.
     *
     * @param autogradModeFor A predicate lambda which determines the auto diff mode of this algorithm a given execution call.
     * @return This very instance to enable method chaining.
     */
    public final AbstractFunAlgorithm setAutogradModeFor(ADSupportPredicate autogradModeFor ) {
        _autogradModeFor = _checked(autogradModeFor, _autogradModeFor, ADSupportPredicate.class);
        return this;
    }

    @Override
    public AutoDiffMode autoDiffModeFrom(ExecutionCall<? extends Device<?>> call ) {
        _checkReadiness();
        return _autogradModeFor.autoDiffModeFrom( call );
    }

    public final AbstractFunAlgorithm setExecution(Execution execution ) {
        _execution = _checked(execution, _execution, Execution.class);
        return this;
    }

    @Override
    public Result execute( Function caller, ExecutionCall<? extends Device<?>> call ) {
        _checkReadiness();
        MemValidator checker = MemValidator.forInputs( call.inputs(), ()-> _execution.execute( caller, call ));
        if ( checker.isWronglyIntermediate() ) {
            throw new IllegalStateException(
                    "Output of algorithm '" + this.getName() + "' " +
                    "is marked as intermediate result, despite the fact " +
                    "that it is a member of the input array. " +
                    "Tensors instantiated by library users instead of operations in the backend are not supposed to be flagged " +
                    "as 'intermediate', because they are not eligible for deletion!"
            );
        }
        if ( checker.isWronglyNonIntermediate() ) {
            throw new IllegalStateException(
                    "Output of algorithm '" + this.getName() + "' " +
                    "is neither marked as intermediate result nor a member of the input array. " +
                    "Tensors instantiated by operations in the backend are expected to be flagged " +
                    "as 'intermediate' in order to be eligible for deletion!"
            );
        }
        return checker.getResult();
    }

}
