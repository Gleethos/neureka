package neureka.autograd;

import lombok.Setter;
import lombok.experimental.Accessors;
import neureka.Tsr;

import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;


/**
 * This class implements the ADAgent interface.
 * ADAgent stands for "Auto-Differentiation-Agent", meaning
 * that implementations of this class are responsible for managing
 * forward- and reverse- mode differentiation actions.
 * These actions are accessible through the "forward(...)"
 * and "backward(...)" method which are being triggered
 * by instances of the GraphNode class during propagation.
 *
 * This class stores implementations for these methods
 * inside the agent as lambda instances.
 *
 * So in essence this class is a container for lambda actions
 * allowing for easy instantiation of ADAgents.
 * Additionally this class the class manages a variable context
 * for storing useful data used by a particular operation to
 * perform propagation.
 *
 */
@Accessors( prefix = {"_"}, chain = true )
public final class DefaultADAgent implements ADAgent
{
    /**
     * This interface is the declaration for
     * lambda actions for both the "forward(...)"
     * and "backward(...)" method of the ADAgent interface.
     */
    public interface ADAction
    {
         Tsr<?> execute(GraphNode<?> t, Tsr<?> error);
    }

    @Setter private ADAction _forward;
    @Setter private ADAction _backward;
    private final Map<String, Object> _context = new TreeMap<>();

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
        return (
                _context != null &&
                _context.containsKey( "derivative" )
        );
    }

    @Override
    public boolean hasBackward() {
        return _backward != null;
    }

    @Override
    public String toString() {
        if ( this.derivative() != null ) {
            return derivative().toString();
        }

        return _context.keySet().stream()
                .map( key -> key + "=" + _context.get( key ) )
                .collect( Collectors.joining( ", ", "{", "}" ) );
    }










}
