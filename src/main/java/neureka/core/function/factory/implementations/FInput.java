package neureka.core.function.factory.implementations;


import neureka.core.Tsr;
import neureka.core.function.Function;

public class FInput implements Function, IProvider
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
    public Function newBuild(final String equation) {
        int number = 0;
        for (int i = 0; i < equation.length(); ++i) {
            if (equation.charAt(i) == 'j') {
                Function newCore = new FVariable();
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
    public double activate(final double[] inputs, int j) {
        return inputs[_index()];
    }
    @Override
    public double activate(final double[] inputs) {
        return inputs[(_index>=0)?_index:(Math.abs(_index)-1)];
    }
    @Override
    public double derive(final double[] inputs, final int index) {
        return (index == _index()) ? 1 : 0;
    }
    @Override
    public double derive(double[] inputs, int index, int j) {
        return derive(inputs, index);
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public Tsr activate(Tsr[] inputs, int j) {
        if(this.providesGradient() && inputs[_index()].rqsGradient()){
            inputs[_index()].setGradientIsTargeted(true);
        }
        return inputs[_index()];
    }
    @Override
    public Tsr activate(Tsr[] inputs) {
        if(this.providesGradient() && inputs[_index()].rqsGradient()){
            inputs[_index()].setGradientIsTargeted(true);
        }
        return inputs[_index()];
    }
    @Override
    public Tsr derive(Tsr[] inputs, int index, int j) {
        return derive(inputs, index);
    }
    @Override
    public Tsr derive(Tsr[] inputs, int index) {
        return (index == _index())
                ? Tsr.fcn.create.newTsr(1, inputs[0].shape())
                : Tsr.fcn.create.newTsr(0, inputs[0].shape());
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
