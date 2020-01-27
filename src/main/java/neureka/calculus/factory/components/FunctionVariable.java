package neureka.calculus.factory.components;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.autograd.GraphNode;
import neureka.calculus.Function;
import neureka.calculus.factory.assembly.FunctionBuilder;

public class FunctionVariable implements Function, GradientProvider {

    private boolean _providesGradient = false;

    public boolean providesGradient(){
        return _providesGradient;
    }

    @Override
    public boolean isFlat() {
        return false;
    }

    @Override
    public boolean doesAD(){
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

    @Override
    public boolean dependsOn(int index){
        return true;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public Function newBuild(final String equation) {
        if(equation.contains("g")){
            _providesGradient = true;
        }
        return this;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double activate(final double[] inputs, int j) {
        return inputs[j];
    }
    @Override
    public double activate(final double[] inputs) {
        double sum = 0;
        for (int Ii = 0; Ii < inputs.length; Ii++) {
            sum += activate(inputs, Ii);
        }
        return sum;
    }
    @Override
    public double derive(final double[] inputs, final int index) {
        return 1.0;
    }
    @Override
    public double derive(double[] inputs, int index, int j) {
        if (j != index) {
            return 0;
        }
        return derive(inputs, index);
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public Tsr activate(Tsr[] inputs, int j) {
        return inputs[j];
    }
    @Override
    public Tsr activate(Tsr[] inputs) {
        String exp = "I[0]";
        for(int i=1; i<inputs.length; i++)exp += "+I["+i+"]";
        return FunctionBuilder.build(exp, false).activate(inputs);
    }
    @Override
    public Tsr derive(Tsr[] inputs, int index, int j) {
        if (j != index) {
            return new Tsr(inputs[0].shape(), 0.0);
        }
        return derive(inputs, index);
    }
    @Override
    public Tsr derive(Tsr[] inputs, int index) {
        return new Tsr(inputs[0].shape(), 1.0);
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public String toString() {
        return "I"+((this.providesGradient())?"g":"")+"[j]";
    }

    @Override
    public ADAgent getReverseAD(GraphNode node, Tsr[] inputs, int i){
        return null;
    }

    //@Override
    //public FADLambda getForwardAD(GraphNode node, Tsr[] inputs, int i){
    //    return (error)->error;
    //}


}
