package neureka.core.function.factory.implementations;

import neureka.core.T;
import neureka.core.function.IFunction;

public class FVariable implements IFunction, IProvider {

    private boolean _providesGradient = false;

    public boolean providesGradient(){
        return _providesGradient;
    }

    @Override
    public boolean isFlat() {
        return false;
    }
    @Override
    public int id() {
        return -1;
    }

    @Override
    public String type() {
        return "variable";
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public IFunction newBuild(final String equation) {
        if(equation.contains("g")){
            _providesGradient = true;
        }
        return this;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double activate(final double[] input, int j) {
        return input[j];
    }
    @Override
    public double activate(final double[] input) {
        double sum = 0;
        for (int Ii = 0; Ii < input.length; Ii++) {
            sum += activate(input, Ii);
        }
        return sum;
    }
    @Override
    public double derive(final double[] input, final int index) {
        return 1.0;
    }
    @Override
    public double derive(double[] input, int index, int j) {
        if (j != index) {
            return 0;
        }
        return derive(input, index);
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public T activate(T[] input, int j) {
        if(this.providesGradient() && input[j].rqsGradient()){
            input[j].setGradientIsTargeted(true);
        }
        return input[j];
    }
    @Override
    public T activate(T[] input) {
        return new T(input, "sum(I[j])");
    }
    @Override
    public T derive(T[] input, int index, int j) {
        if (j != index) {
            return T.factory.newTensor(0, input[0].shape());
        }
        return derive(input, index);
    }
    @Override
    public T derive(T[] input, int index) {
        return T.factory.newTensor(1, input[0].shape());
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public String toString() {
        return "I"+((this.providesGradient())?"g":"")+"[j]";
    }
}
