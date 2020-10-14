package neureka.autograd;

import neureka.Tsr;
import neureka.calculus.Function;

import java.util.Map;
import java.util.TreeMap;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class ADAgent
{
    public interface ADAction
    {
        Tsr execute(GraphNode t, Tsr error);
    }

    private ADAction _fad;
    private ADAction _bad;
    private Map<String, Object> _context = new TreeMap<>();

    public ADAgent(  Tsr<?> derivative  ){
        _context.put( "derivative", derivative );
    }

    public ADAgent(){ }

    public ADAgent withContext(  Map<String, Object> context  ) {
        _context.putAll( context );
        return this;
    }

    public ADAgent withForward(ADAction fad) {
        _fad = fad;
        return this;
    }

    public ADAgent withBackward(ADAction bad){
        _bad = bad;
        return this;
    }

    public Tsr forward(GraphNode t, Tsr error){
        return _fad.execute(t, error);
    }

    public Tsr backward(GraphNode t, Tsr error){
        return _bad.execute(t, error);
    }

    public Tsr derivative(){
        return (Tsr) _context.get("derivative");
    }

    public boolean isForward(){
        return (_bad==null);
    }

    @Override
    public String toString(){
        if(this.derivative()!=null){
            return derivative().toString();
        }
        if(this.isForward()){

        }
        if ( _context != null ) {
            String mapAsString = _context.keySet().stream()
                    .map(key -> key + "=" + _context.get(key))
                    .collect(Collectors.joining(", ", "{", "}"));
            return mapAsString;
        }
        return "null";
    }










}
