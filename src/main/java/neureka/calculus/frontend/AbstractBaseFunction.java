package neureka.calculus.frontend;

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
public abstract class AbstractBaseFunction implements Function
{

    @Override
    public double call(double input){
        return call(new double[]{input});
    }

    @Override
    public <T> Tsr<T> call(Tsr<T> input){
        return call(new Tsr[]{input});
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public <T> Tsr<T> call( List<Tsr<T>> inputs ) {
        return call(inputs.toArray(new Tsr[ 0 ]));
    }

    @Override
    public <T> Tsr<T> invoke(List<Tsr<T>> inputs) {
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
    public <T> Tsr<T> invoke(Tsr<T> input){
        return call(input);
    }

    @Override
    public <T> Tsr<T> invoke(Tsr<T>[] inputs, int j){
        return call(inputs, j);
    }

    @Override
    public <T> Tsr<T> invoke(Tsr<T>[] inputs){
        return call(inputs);
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public <T> Tsr<T> derive(List<Tsr<T>> inputs, int index, int j){
        return derive(inputs.toArray(new Tsr[ 0 ]), index, j);
    }

    @Override
    public <T> Tsr<T> derive(List<Tsr<T>> inputs, int index){
        return derive(inputs.toArray(new Tsr[ 0 ]), index);
    }


}
