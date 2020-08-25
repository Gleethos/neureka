package neureka.autograd;

import neureka.Tsr;
import neureka.calculus.Function;

import java.util.function.Supplier;

public class ADAgent
{
    public interface ADAction
    {
        Tsr execute(GraphNode t, Tsr error);
    }

    private ADAction _fad;
    private ADAction _bad;
    private Supplier<Tsr> _derivative;

    public ADAgent(  Supplier<Tsr> derivative  ){
        _derivative = derivative;
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
        return _derivative.get();
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
        return "null";
    }










}
