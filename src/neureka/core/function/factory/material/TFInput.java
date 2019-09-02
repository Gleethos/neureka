package neureka.core.function.factory.material;


import neureka.core.T;
import neureka.core.function.TFunction;

public class TFInput implements TFunction
{
    int _index;
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
    public TFunction newBuild(final String equation) {
        int number = 0;
        for (int i = 0; i < equation.length(); ++i) {
            if (equation.charAt(i) == 'j') {
                TFunction newCore = new TFVariable();
                newCore = newCore.newBuild(equation);
                return newCore;
            }
            if (equation.charAt(i) <= '9' && equation.charAt(i) >= '0') {
                number *= 10;
                number += Integer.parseInt(equation.charAt(i) + "");
            }
        }
        this._index = number;
        return this;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double activate(final double[] input, int j) {
        return input[_index];
    }
    @Override
    public double activate(final double[] input) {
        return input[_index];
    }
    @Override
    public double derive(final double[] input, final int index) {
        return (index == this._index) ? 1 : 0;
    }
    @Override
    public double derive(double[] input, int index, int j) {
        return derive(input, index);
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public T activate(T[] input, int j) {
        return input[_index];
    }
    @Override
    public T activate(T[] input) {
        return input[_index];
    }
    @Override
    public T derive(T[] input, int index, int j) {
        return derive(input, index);
    }
    @Override
    public T derive(T[] input, int index) {
        return (index == this._index)
                ? T.factory.newTensor(1, input[0].shape())
                : T.factory.newTensor(0, input[0].shape());
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public String toString() {
        return "I[" + this._index + "]";
    }

}
