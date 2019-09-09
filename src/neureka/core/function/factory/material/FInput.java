package neureka.core.function.factory.material;


import neureka.core.T;
import neureka.core.function.IFunction;

public class FInput implements IFunction, IProvider
{
    private int _index;

    public boolean providesGradient(){
        return (_index<0);
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public boolean isFlat() {
        return true;
    }
    @Override
    public int id() {
        return -1;
    }

    @Override
    public String type() {
        return "input";
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public IFunction newBuild(final String equation) {
        int number = 0;
        for (int i = 0; i < equation.length(); ++i) {
            if (equation.charAt(i) == 'j') {
                IFunction newCore = new FVariable();
                newCore = newCore.newBuild(equation);
                return newCore;
            }
            if (equation.charAt(i) <= '9' && equation.charAt(i) >= '0') {
                number *= 10;
                number += Integer.parseInt(equation.charAt(i) + "");
            }
        }
        _index = number;
        if(equation.contains("g")){
            _index = -(_index+1);
        }

        return this;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double activate(final double[] input, int j) {
        return input[_index()];
    }
    @Override
    public double activate(final double[] input) {
        return input[(_index>=0)?_index:(Math.abs(_index)-1)];
    }
    @Override
    public double derive(final double[] input, final int index) {
        return (index == _index()) ? 1 : 0;
    }
    @Override
    public double derive(double[] input, int index, int j) {
        return derive(input, index);
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public T activate(T[] input, int j) {
        if(this.providesGradient() && input[_index()].rqsGradient()){
            input[_index()].setGradientIsTargeted(true);
        }
        return input[_index()];
    }
    @Override
    public T activate(T[] input) {
        if(this.providesGradient() && input[_index()].rqsGradient()){
            input[_index()].setGradientIsTargeted(true);
        }
        return input[_index()];
    }
    @Override
    public T derive(T[] input, int index, int j) {
        return derive(input, index);
    }
    @Override
    public T derive(T[] input, int index) {
        return (index == _index())
                ? T.factory.newTensor(1, input[0].shape())
                : T.factory.newTensor(0, input[0].shape());
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public String toString() {
        return "I"+((this.providesGradient())?"g":"")+"[" + _index() + "]";
    }

    private int _index(){
        return ((this.providesGradient())?(Math.abs(_index)-1):_index);
    }

}
