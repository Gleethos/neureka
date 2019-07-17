package neureka.main.core.modul.calc;

import neureka.main.core.base.data.T;

public class Function_Base implements Function{

    protected int type;

    Function_Base(int f_id){
        type = f_id;
    }

    @Override
    public boolean isFlat(){
        return  false;
    }
    @Override
    public Function newBuild(String expression){
        return new FunctionConstructor().newBuild(expression);
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public String toString() {
        return "";
    }

    @Override
    public T activate(T[] input, int j) { return null;
    }

    @Override
    public T activate(T[] input) {
        return null;
    }

    @Override
    public T derive(T[] input, int index, int j) {
        return null;
    }

    @Override
    public T derive(T[] input, int index) {
        return null;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double activate(final double[] input, int j) {
        return 0;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double activate(final double[] input) {
        return 0;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double derive(final double[] input, final int index, final int j) {
        return 0;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double derive(final double[] input, final int index) {
        return 0;
    }

}
