package neureka.core.module.calc.fcomp;

import neureka.core.T;
import neureka.core.module.calc.FunctionFactory;
import neureka.core.module.calc.fcomp.util.Context;

import java.util.ArrayList;

public abstract class Template implements Function {

    protected int f_id;
    protected boolean isFlat;
    ArrayList<Function> Srcs;

    protected Template(int f_id, boolean isFlat, ArrayList<Function> Srcs){
        this.f_id = f_id;
        this.isFlat = isFlat;
        this.Srcs = Srcs;
    }

    @Override
    public boolean isFlat(){
        return  this.isFlat;
    }

    @Override
    public int id() {
        return f_id;
    }

    @Override
    public Function newBuild(String expression){
        return FunctionFactory.newBuild(expression);
    }

    @Override
    public String toString() {
        String reconstructed = "";
        if (Srcs.size() == 1 && Context.REGISTER[f_id].length() > 1) {
            String expression = Srcs.get(0).toString();
            if (expression.charAt(0) == '(' && expression.charAt(expression.length() - 1) == ')') {
                return Context.REGISTER[f_id] + expression;
            }
            return Context.REGISTER[f_id] + "(" + expression + ")";
        }
        for (int i = 0; i < Srcs.size(); ++i) {
            if (Srcs.get(i) != null) {
                reconstructed = reconstructed + Srcs.get(i).toString();
            } else {
                reconstructed = reconstructed + "(null)";
            }
            if (i != Srcs.size() - 1) {
                reconstructed = reconstructed + Context.REGISTER[f_id];
            }
        }
        return "(" + reconstructed + ")";
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public abstract T activate(T[] input, int j);

    @Override
    public abstract T activate(T[] input);

    @Override
    public abstract T derive(T[] input, int index, int j);

    @Override
    public abstract T derive(T[] input, int index);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public abstract double activate(final double[] input, int j);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public abstract double activate(final double[] input);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public abstract double derive(final double[] input, final int index, final int j);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public abstract double derive(final double[] input, final int index);

}
