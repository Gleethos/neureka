package neureka.calculus.factory;

import neureka.Tsr;
import neureka.calculus.Function;
import java.util.List;

public abstract class BaseFunction implements Function {

    @Override
    public double call(double input){
        return call(new double[]{input});
    }

    @Override
    public Tsr call(Tsr input){
        return call(new Tsr[]{input});
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public Tsr call(List<Tsr> inputs) {
        return call(inputs.toArray(new Tsr[0]));
    }

    @Override
    public Tsr invoke(List<Tsr> inputs) {
        return call(inputs);
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public double invoke(double input){
        return call(input);
    }

    @Override
    public double invoke(double[] inputs, int j){
        return call(inputs, j);
    }

    @Override
    public double invoke(double[] inputs){
        return call(inputs);
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public Tsr invoke(Tsr input){
        return call(input);
    }

    @Override
    public Tsr invoke(Tsr[] inputs, int j){
        return call(inputs, j);
    }

    @Override
    public Tsr invoke(Tsr[] inputs){
        return call(inputs);
    }


}
