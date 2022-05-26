package neureka.autograd;


import neureka.Tsr;

import java.util.Optional;


/**
 *  {@link ADAgent} stands for "Auto-Differentiation-Agent", meaning
 *  that implementations of this class are responsible for managing
 *  forward- and reverse- mode differentiation actions.
 *  These differentiation actions are performed through the "{@link ADAgent#act(Target)}"
 *  method which are being called
 *  by instances of the {@link GraphNode} class during propagation.
 *  An {@link ADAgent} may also wrap and expose a partial derivative
 *  which may or may not be present for certain operations.
 *  <br>
 *  This class stores implementations for the propagation method
 *  inside the agent as a lambda instance. <br>
 *
 *  So in essence this class is a container for a lambda as well as an optional derivative.
 */
public final class DefaultADAgent implements ADAgent {

    public static WithAD ofDerivative( Tsr<?> derivative ) {
        return action -> new DefaultADAgent( derivative, action );
    }
    
    /**
     *  This lambda ought to perform the forward or backward propagation
     *  for the concrete {@link neureka.backend.api.ImplementationFor} of a {@link neureka.devices.Device}.
     */
    private final ADAction _action;
    private final Tsr<?> _partialDerivative;

    /**
     * @param derivative The current derivative which will be stored with the name "derivative" in the agents context.
     */
    private DefaultADAgent( Tsr<?> derivative, ADAction action ) { _partialDerivative = derivative; _action = action; }

    @Override
    public <T> Tsr<T> act( Target<T> target ) {
        if ( _action == null )
            throw new IllegalStateException(
                "Cannot perform propagation because this "+ADAgent.class.getSimpleName()+" does have an auto-diff implementation."
            );
        return (Tsr<T>) _action.act( target );
    }

    @Override
    public Optional<Tsr<?>> partialDerivative() { return Optional.ofNullable(_partialDerivative); }

    /**
     *  An {@link ADAgent} also contains a context of variables which have been
     *  passed to it by an {@link neureka.backend.api.ExecutionCall}.
     *  A given {@link neureka.backend.api.ExecutionCall} will itself have gathered the context
     *  variables within a given backend implementation, more specifically an {@link neureka.backend.api.Operation}.
     *  These variables are used by an implementation of the {@link neureka.backend.api.Operation} to perform auto differentiation
     *  or to facilitate further configuration of an {@link neureka.backend.api.ExecutionCall}.
     *  This method let's us view the current state of these variables for this agent in the form of
     *  a nice {@link String}...
     *
     * @return A String view of this {@link ADAgent}.
     */
    @Override
    public String toString() {
        if ( this.partialDerivative().isPresent() ) return partialDerivative().get().toString();
        return "";
    }

    public interface WithAD {
        ADAgent withAD( ADAction action );
    }

    /**
     * This interface is the declaration for
     * lambda actions for both the {@link #act(Target)} method of the {@link ADAgent} interface. <br><br>
     *
     * Note: Do not access the {@link GraphNode#getPayload()} of the {@link GraphNode}
     *       passed to implementation of this.
     *       The payload is weakly referenced, meaning that this method can return null!
     */
    public interface ADAction
    {
        /**
         * @param target A container for the {@link GraphNode} at which the differentiation ought to be performed and error which ought to be used for the forward or backward differentiation.
         * @return The result of the differentiation.
         */
        Tsr<?> act( Target<?> target );
    }

}
