package neureka.autograd;


import neureka.Tsr;

import java.util.Optional;


/**
 *  {@link ADAgent} stands for "Auto-Differentiation-Agent", meaning
 *  that implementations of this class are responsible for managing
 *  forward- and reverse- mode differentiation actions.
 *  These differentiation actions are performed through the "{@link ADAgent#act(ADTarget)}"
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
final class DefaultADAgent implements ADAgent
{
    /**
     *  This lambda ought to perform the forward or backward propagation
     *  for the concrete {@link neureka.backend.api.ImplementationFor} of a {@link neureka.devices.Device}.
     */
    private final ADAction _action;

    DefaultADAgent( ADAction action ) { _action = action; }

    @Override
    public Tsr<?> act( ADTarget<?> target ) {
        if ( _action == null )
            throw new IllegalStateException(
                "Cannot perform propagation because this "+ADAgent.class.getSimpleName()+" does have an auto-diff implementation."
            );
        return _action.act( target );
    }

    @Override
    public Optional<Tsr<?>> partialDerivative() {
        Tsr<?>[] captured = _action.findCaptured();
        if ( captured.length > 0 )
            return Optional.of(captured[captured.length - 1]);

        return Optional.empty();
    }

    /**
     *  An {@link ADAgent} also contains a context of variables which have been
     *  passed to it by an {@link neureka.backend.api.ExecutionCall}.
     *  A given {@link neureka.backend.api.ExecutionCall} will itself have gathered the context
     *  variables within a given backend implementation, more specifically an {@link neureka.backend.api.Operation}.
     *  These variables are used by an implementation of the {@link neureka.backend.api.Operation} to perform auto differentiation
     *  or to facilitate further configuration of an {@link neureka.backend.api.ExecutionCall}.
     *  This method lets us view the current state of these variables for this agent in the form of
     *  a nice {@link String}...
     *
     * @return A String view of this {@link ADAgent}.
     */
    @Override
    public String toString() {
        if ( this.partialDerivative().isPresent() ) return partialDerivative().get().toString();
        return "";
    }

}
