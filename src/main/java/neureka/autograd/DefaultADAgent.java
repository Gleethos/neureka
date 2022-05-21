package neureka.autograd;


import neureka.Tsr;
import neureka.calculus.args.Arg;
import neureka.calculus.args.Args;

import java.util.List;
import java.util.stream.Collectors;


/**
 *  This class implements the ADAgent interface.
 *  {@link ADAgent} stands for "Auto-Differentiation-Agent", meaning
 *  that implementations of this class are responsible for managing
 *  forward- and reverse- mode differentiation actions.
 *  These actions are accessible through the {@link #act(GraphNode, Tsr)}
 *  and {@link #act(GraphNode, Tsr)} method which are being triggered
 *  by instances of the GraphNode class during propagation. <br>
 *  <br>
 *  This class stores implementations for these methods
 *  inside the agent as lambda instances. <br>
 *
 *  So in essence this class is a container for lambda actions
 *  allowing for easy instantiation of {@link ADAgent}s.
 *  Additionally this class the class manages a variable context
 *  for storing useful data used by a particular {@link neureka.backend.api.Operation} to
 *  perform propagation. <br>
 *  This context will be populated by a given {@link neureka.backend.api.ExecutionCall},
 *  which will itself have gathered the context
 *  variables within a given backend implementation, more specifically an {@link neureka.backend.api.Operation}.
 *  These variables are used by an implementation of the {@link neureka.backend.api.Operation} to perform auto differentiation
 *  or to facilitate further configuration of an {@link neureka.backend.api.ExecutionCall}.
 */
public final class DefaultADAgent implements ADAgent {

    public static DefaultADAgent ofDerivative( Tsr<?> derivative ) { return new DefaultADAgent( derivative ); }
    
    /**
     *  This lambda ought to perform the forward or backward propagation
     *  for the concrete {@link neureka.backend.api.ImplementationFor} of a {@link neureka.devices.Device}.
     */
    private ADAction _action = (node, forwardDerivative ) -> {throw new IllegalStateException("Forward AD not defined!");};
    private final Tsr<?> _partialDerivative;

    /**
     * @param derivative The current derivative which will be stored with the name "derivative" in the agents context.
     */
    private DefaultADAgent( Tsr<?> derivative ) { _partialDerivative = derivative; }


    public DefaultADAgent setAction( ADAction action ) { _action = action; return this; }

    @Override
    public <T> Tsr<T> act(GraphNode<T> target, Tsr<T> derivativeOrError) { return (Tsr<T>) _action.execute( target, derivativeOrError); }

    @Override
    public Tsr<?> partialDerivative() {
        return _partialDerivative;
    }

    @Override
    public boolean hasAction() { return _action != null; }

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
        if ( this.partialDerivative() != null ) return partialDerivative().toString();
        return "";//getAll(Arg.class).stream()
        //        .map( key -> key.getClass().getSimpleName() + "=" + get(key.getClass()) )
        //        .collect( Collectors.joining( ", ", "{", "}" ) );
    }


    /**
     * This interface is the declaration for
     * lambda actions for both the {@link #act(GraphNode, Tsr)}
     * and {@link #act(GraphNode, Tsr)} method of the {@link ADAgent} interface. <br><br>
     *
     * Note: Do not access the {@link GraphNode#getPayload()} of the {@link GraphNode}
     *       passed to implementation of this.
     *       The payload is weakly referenced, meaning that this method can return null!
     */
    public interface ADAction
    {
        /**
         * @param node The {@link GraphNode} at which the differentiation ought to be performed.
         * @param error The error which ought to be used for the forward or backward differentiation.
         * @return The result of the differentiation.
         */
        Tsr<?> execute( GraphNode<?> node, Tsr<?> error );
    }

    
}
