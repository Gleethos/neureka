package neureka.calculus.implementations;


import neureka.Tsr;
import neureka.backend.api.operations.AbstractOperation;
import neureka.calculus.Function;
import neureka.calculus.AbstractBaseFunction;
import neureka.calculus.assembly.FunctionBuilder;

/**
 *  Instances of this implementation of the {@link Function} interface
 *  are leave nodes within the abstract syntax tree of a function, representing inputs to a function.
 *  When parsing an expression into a function then these inputs are recognized by the character 'i' or 'I',
 *  followed by a whole number starting at zero (optionally wrapped by '[' & ']'), which is the index
 *  of the argument within the list/array of arguments passed to a concrete {@link Function} instance. <br>
 *  So for example, when creating a function by calling the following factory method...     <br>
 *                                                                                          <br>
 *  {@link Function#create}( "I[1] + (4 * I[0]) / 2" )                                      <br>
 *                                                                                          <br>
 *  ...then the substrings "I[1]" and "I[0]" will be parsed into instances of this class!   <br>
 *  When calling this function by passing two arguments, let's say (first, second) then
 *  the {@link FunctionInput} "I[0]" will pick the first argument, whereas "I[1]"
 *  will pick the second argument when evaluating the array of arguments.
 *
 */
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
    public AbstractOperation getOperation() {
        return null;
    }

    @Override
    public boolean dependsOn( int index ) {
        return index() == index;
    }

    @Override
    public Function getDerivative( int index ) {
        return ( index == _index ) ? Function.create( "1" ) : Function.create( "0" );
    }

    //------------------------------------------------------------------------------------------------------------------

    @Override
    public Function newBuild(final String equation) {

        if (equation.charAt( 0 )=='-') {
            return FunctionBuilder.build(equation.substring(1)+"*-1", true); // TODO: This might be false!
        }
        int number = 0;
        for ( int i = 0; i < equation.length(); ++i) {
            if (equation.charAt( i ) <= '9' && equation.charAt( i ) >= '0') {
                number *= 10;
                number += Integer.parseInt(equation.charAt( i ) + "");
            }
        }
        _index = number;
        if (equation.contains("g")) {
            _index = -(_index + 1 );
        }

        return this;
    }

    private Tsr<?> _extract(Tsr<?> t)
    {
        if (this.providesGradient() && t.rqsGradient()) {
            Tsr<?> gradient = t.getGradient();
            if (t.rqsGradient()) {
                if (gradient==null) {
                    gradient = new Tsr<>(t.shape(), 0);
                    t.set((Tsr)gradient);
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
    public double call(final double... inputs) {
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
    public Tsr<?> execute(Tsr<?>[] inputs, int j) {
        return _extract(inputs[index()]);
    }

    @Override
    public Tsr<?> execute(Tsr<?>... inputs) {
        return _extract(inputs[index()]);
    }

    @Override
    public Tsr<?> executeDerive(Tsr<?>[] inputs, int index, int j) {
        return executeDerive(inputs, index);
    }

    @Override
    public Tsr<?> executeDerive(Tsr<?>[] inputs, int index) {
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
