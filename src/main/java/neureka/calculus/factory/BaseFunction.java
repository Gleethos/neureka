package neureka.calculus.factory;

import neureka.Tsr;
import neureka.calculus.Function;
import java.util.List;

/**
 * This class implements certain methods by simply calling
 * other methods which are responsible for executing
 * the important logic implemented in sub classes.
 *
 * The reason for this is simply that otherwise there
 * would be many redundantly implemented methods.
 * The 'call' and 'invoke' methods with the same arguments
 * are supposed to do the same thing, however
 * they are both part of the Function interface in order to
 * allow for overloading the '()' operator in different
 * JVM languages...
 */
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

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public Tsr derive(List<Tsr> inputs, int index, int j){
        return derive(inputs.toArray(new Tsr[0]), index, j);
    }

    @Override
    public Tsr derive(List<Tsr> inputs, int index){
        return derive(inputs.toArray(new Tsr[0]), index);
    }


}
