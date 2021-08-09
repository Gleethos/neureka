package neureka.autograd;


import neureka.Tsr;

import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;


/**
 *  This class implements the ADAgent interface.
 *  {@link ADAgent} stands for "Auto-Differentiation-Agent", meaning
 *  that implementations of this class are responsible for managing
 *  forward- and reverse- mode differentiation actions.
 *  These actions are accessible through the {@link #forward(GraphNode, Tsr)}
 *  and {@link #backward(GraphNode, Tsr)} method which are being triggered
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
public final class DefaultADAgent implements ADAgent
{
    public DefaultADAgent setForward( ADAction _forward ) { this._forward = _forward; return this; }

    public DefaultADAgent setBackward( ADAction _backward ) { this._backward = _backward; return this; }

    /**
     * This interface is the declaration for
     * lambda actions for both the {@link #forward(GraphNode, Tsr)}
     * and {@link #backward(GraphNode, Tsr)} method of the {@link ADAgent} interface.
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

    /**
     *  This lambda ought to perform the forward propagation
     *  for the concrete {@link neureka.backend.api.ImplementationFor} of a {@link neureka.devices.Device}.
     */
    private ADAction _forward;
    /**
     *  This lambda ought to perform the backward propagation
     *  for the concrete {@link neureka.backend.api.ImplementationFor} of a {@link neureka.devices.Device}.
     */
    private ADAction _backward;
    /**
     *  A map of named context variables which can be used by any backend implementation to
     *  save variables useful to perform differentiation.
     */
    private final Map<String, Object> _context = new TreeMap<>();

    /**
     * @param derivative The current derivative which will be stored with the name "derivative" in the agents context.
     */
    public DefaultADAgent(  Tsr<?> derivative  ) {
        _context.put( "derivative", derivative );
    }

    public DefaultADAgent() { }

    public DefaultADAgent withContext( Map<String, Object> context  ) {
        _context.putAll( context );
        return this;
    }

    @Override
    public <T> Tsr<T> forward( GraphNode<T> target, Tsr<T> derivative) {
        return (Tsr<T>) _forward.execute( target, derivative);
    }

    @Override
    public <T> Tsr<T> backward( GraphNode<T> target, Tsr<T> error ) {
        return (Tsr<T>) _backward.execute( target, error );
    }

    @Override
    public Tsr<?> derivative() {
        return (Tsr<?>) _context.get( "derivative" );
    }

    @Override
    public boolean hasForward() {
        return _context.containsKey( "derivative" );
    }

    @Override
    public boolean hasBackward() {
        return _backward != null;
    }

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
        if ( this.derivative() != null ) return derivative().toString();
        return _context.keySet().stream()
                .map( key -> key + "=" + _context.get( key ) )
                .collect( Collectors.joining( ", ", "{", "}" ) );
    }










}
