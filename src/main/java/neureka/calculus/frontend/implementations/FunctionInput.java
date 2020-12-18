package neureka.calculus.frontend.implementations;


import neureka.Tsr;
import neureka.calculus.Function;
import neureka.calculus.backend.operations.AbstractOperationType;
import neureka.calculus.frontend.AbstractBaseFunction;
import neureka.calculus.frontend.assembly.FunctionBuilder;

public class FunctionInput extends AbstractBaseFunction implements GradientProvider
{
    private int _index;

    //------------------------------------------------------------------------------------------------------------------

    public boolean providesGradient() {
        return (_index<0);
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public boolean isFlat() {
        return true;
    }

    @Override
    public boolean isDoingAD() {
        return false;
    }

    @Override
    public AbstractOperationType getOperation() {
        return null;
    }

    @Override
    public boolean dependsOn(int index) {
        return index() == index;
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public Function newBuild(final String equation) {

        if(equation.charAt( 0 )=='-') {
            return FunctionBuilder.build(equation.substring(1)+"*-1", true);
        }
        int number = 0;
        for (int i = 0; i < equation.length(); ++i) {
            if (equation.charAt( i ) == 'j') {
                Function newCore = new FunctionVariable();
                newCore = newCore.newBuild(equation);
                return newCore;
            }
            if (equation.charAt( i ) <= '9' && equation.charAt( i ) >= '0') {
                number *= 10;
                number += Integer.parseInt(equation.charAt( i ) + "");
            }
        }
        _index = number;
        if(equation.contains("g")) {
            _index = -(_index+1);
        }

        return this;
    }

    private Tsr _extract(Tsr t)
    {
        if (this.providesGradient() && t.rqsGradient()) {
            Tsr gradient = (Tsr) t.find(Tsr.class);
            if (t.rqsGradient()) {
                if (gradient==null) {
                    gradient = new Tsr(t.shape(), 0);
                    t.set(gradient);
                }
                return gradient;
            }
        }
        return t;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public double call(final double[] inputs, int j) {
        return inputs[index()];
    }

    @Override
    public double call(final double[] inputs) {
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

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public Tsr call(Tsr[] inputs, int j) {
        return _extract(inputs[index()]);
    }

    @Override
    public Tsr call(Tsr[] inputs) {
        return _extract(inputs[index()]);
    }

    @Override
    public Tsr derive(Tsr[] inputs, int index, int j) {
        return derive(inputs, index);
    }

    @Override
    public Tsr derive(Tsr[] inputs, int index) {
        return ( index == index() )
                ? new Tsr(inputs[ 0 ].shape(), 1.0)
                : new Tsr(inputs[ 0 ].shape(), 0.0);
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public String toString() {
        return "I"+((this.providesGradient())?"g":"")+"[" + index() + "]";
    }

    public int index() {
        return ((this.providesGradient())?(Math.abs(_index)-1):_index);
    }

}
