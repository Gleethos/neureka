package neureka.calculus.implementations;

import neureka.Tsr;
import neureka.calculus.Function;
import neureka.backend.api.operations.AbstractOperation;
import neureka.calculus.AbstractBaseFunction;
import neureka.calculus.assembly.FunctionBuilder;

public class FunctionVariable extends AbstractBaseFunction implements GradientProvider {

    private boolean _providesGradient = false;

    public boolean providesGradient() {
        return _providesGradient;
    }

    @Override
    public boolean isFlat() {
        return false;
    }

    @Override
    public boolean isDoingAD() {
        return false;
    }

    @Override
    public AbstractOperation getOperation() {
        return null;
    }

    @Override
    public boolean dependsOn(int index) {
        return true;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public Function newBuild(final String equation) {
        if (equation.contains("g")) _providesGradient = true;
        return this;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public double call(final double[] inputs, int j) {
        return inputs[j];
    }

    @Override
    public double call(final double[] inputs) {
        double sum = 0;
        for (int Ii = 0; Ii < inputs.length; Ii++) sum += call(inputs, Ii);
        return sum;
    }

    @Override
    public double derive(final double[] inputs, final int index) {
        return 1.0;
    }

    @Override
    public double derive(double[] inputs, int index, int j) {
        if (j != index) return 0;
        return derive(inputs, index);
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public <T> Tsr<T> call(Tsr<T>[] inputs, int j) {
        return inputs[j];
    }

    @Override
    public <T> Tsr<T> call(Tsr<T>[] inputs) {
        String exp = "I[ 0 ]";
        for(int i=1; i<inputs.length; i++)exp += "+I["+i+"]";
        return FunctionBuilder.build(exp, false).call( inputs );
    }

    @Override
    public <T> Tsr<T> derive(Tsr<T>[] inputs, int index, int j) {
        return (j != index) ? new Tsr<T>(inputs[ 0 ].shape(), 0.0) : derive(inputs, index);
    }

    @Override
    public <T> Tsr<T> derive(Tsr<T>[] inputs, int index) {
        return new Tsr<T>(inputs[ 0 ].shape(), 1.0);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public String toString() {
        return "I"+((this.providesGradient())?"g":"")+"[j]";
    }


}
