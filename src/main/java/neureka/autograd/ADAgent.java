package neureka.autograd;

import neureka.Tsr;

import java.util.function.Supplier;

public class ADAgent {

    //public interface ForwardAD{
    //    Tsr execute(Tsr derivative);
    //}

    public interface BackwardAD{
        Tsr execute(GraphNode t, Tsr error);
    }

    private BackwardAD _fad;
    private BackwardAD _bad;
    private Supplier<Tsr> _derivative;

    public ADAgent(
      Supplier<Tsr> derivative, BackwardAD fad, BackwardAD bad
    ){
        _derivative = derivative;
        _fad = fad;
        _bad = bad;
    }
    public ADAgent(
            Tsr derivative
    ){
        _derivative = ()->derivative;
        _fad = null;
        _bad = null;
    }

    public Tsr forward(GraphNode t, Tsr error){
        return _fad.execute(t, error);
        //return _fad.execute(derivative);
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
