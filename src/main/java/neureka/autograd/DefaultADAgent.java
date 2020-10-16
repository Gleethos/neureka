package neureka.autograd;

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
public class DefaultADAgent implements ADAgent
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

    private ADAction _fad;
    private ADAction _bad;
    private final Map<String, Object> _context = new TreeMap<>();

    public DefaultADAgent(  Tsr<?> derivative  ) {
        _context.put( "derivative", derivative );
    }

    public DefaultADAgent(){ }

    public DefaultADAgent withContext(  Map<String, Object> context  ) {
        _context.putAll( context );
        return this;
    }

    public DefaultADAgent withForward(ADAction fad) {
        _fad = fad;
        return this;
    }

    public DefaultADAgent withBackward(ADAction bad){
        _bad = bad;
        return this;
    }

    @Override
    public <T> Tsr<T> forward(GraphNode<T> t, Tsr<T> error){
        return (Tsr<T>) _fad.execute(t, error);
    }

    @Override
    public <T> Tsr<T> backward(GraphNode<T> t, Tsr<T> error){
        return (Tsr<T>) _bad.execute(t, error);
    }

    @Override
    public Tsr<?> derivative(){
        return (Tsr<?>) _context.get("derivative");
    }

    @Override
    public boolean hasForward(){
        return (
                _context != null &&
                _context.containsKey("derivative")
        );
    }

    @Override
    public boolean hasBackward() {
        return _bad != null;
    }

    @Override
    public String toString(){
        if(this.derivative()!=null){
            return derivative().toString();
        }

        if ( _context != null ) {
            return _context.keySet().stream()
                    .map(key -> key + "=" + _context.get(key))
                    .collect(Collectors.joining(", ", "{", "}"));
        }
        return "null";
    }










}
