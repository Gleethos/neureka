package neureka.calculus.factory.components;


import neureka.Tsr;
import neureka.autograd.GraphNode;
import neureka.calculus.Function;
import neureka.calculus.factory.assembly.FunctionBuilder;

public class FunctionInput implements Function, GradientProvider
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
    public boolean doesAD(){
        return false;
    }

    @Override
    public int id() {
        return -1;
    }

    @Override
    public String type() {
        return "input";
    }

    @Override
    public boolean dependsOn(int index){
        if(index()==index) return true;
        return false;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public Function newBuild(final String equation) {

        if(equation.charAt(0)=='-'){
            return FunctionBuilder.build(equation.substring(1)+"*-1", true);
        }
        int number = 0;
        for (int i = 0; i < equation.length(); ++i) {
            if (equation.charAt(i) == 'j') {
                Function newCore = new FunctionVariable();
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

    private Tsr _extract(Tsr t)
    {
        if(this.providesGradient() && t.rqsGradient()){
            Tsr gradient = (Tsr)t.find(Tsr.class);
            if(t.rqsGradient()){
                if(gradient==null){
                    gradient = new Tsr(t.shape(), 0);
                    t.add(gradient);
                }
                return gradient;
            }
        }
        return t;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public double activate(final double[] inputs, int j) {
        return inputs[index()];
    }
    @Override
    public double activate(final double[] inputs) {
        return inputs[(_index>=0)?_index:(Math.abs(_index)-1)];
    }
    @Override
    public double derive(final double[] inputs, final int index) {
        return (index == index()) ? 1 : 0;
    }
    @Override
    public double derive(double[] inputs, int index, int j) {
        return derive(inputs, index);
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public Tsr activate(Tsr[] inputs, int j) {
        return _extract(inputs[index()]);
    }
    @Override
    public Tsr activate(Tsr[] inputs) {
        return _extract(inputs[index()]);
    }
    @Override
    public Tsr derive(Tsr[] inputs, int index, int j) {
        return derive(inputs, index);
    }
    @Override
    public Tsr derive(Tsr[] inputs, int index) {
        return (index == index())
                ? new Tsr(inputs[0].shape(), 1.0)
                : new Tsr(inputs[0].shape(), 0.0);
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public String toString() {
        return "I"+((this.providesGradient())?"g":"")+"[" + index() + "]";
    }

    public int index(){
        return ((this.providesGradient())?(Math.abs(_index)-1):_index);
    }

    @Override
    public RADLambda getReverseAD(GraphNode node, Tsr[] inputs, int i){
        return (error, t)->error;
    }

    //@Override
    //public FADLambda getForwardAD(GraphNode node, Tsr[] inputs, int i){
    //    return (error)->error;
    //}

}
