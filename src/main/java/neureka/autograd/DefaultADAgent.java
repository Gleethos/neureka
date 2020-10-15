package neureka.autograd;

import neureka.Tsr;

import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;


/**
 * ADAgent stands for "Auto-Differentiation-Agent", meaning
 * that instances of this class are responsible for managing
 * forward- and reverse- mode differentiation actions.
 * These actions are stored inside the agent as lambda instances.
 *
 */
public class DefaultADAgent implements ADAgent
{
    public interface ADAction
    {
        Tsr execute(GraphNode t, Tsr error);
    }

    private ADAction _fad;
    private ADAction _bad;
    private Map<String, Object> _context = new TreeMap<>();

    public DefaultADAgent(  Tsr<?> derivative  ){
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
    public <T> Tsr<T> forward(GraphNode t, Tsr<T> error){
        return _fad.execute(t, error);
    }

    @Override
    public <T> Tsr<T> backward(GraphNode t, Tsr<T> error){
        return _bad.execute(t, error);
    }

    @Override
    public Tsr<?> derivative(){
        return (Tsr<?>) _context.get("derivative");
    }

    @Override
    public boolean isForward(){
        return (
                _context != null &&
                _context.containsKey("derivative")//_bad==null
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
