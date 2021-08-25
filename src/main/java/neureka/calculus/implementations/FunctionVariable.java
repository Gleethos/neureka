package neureka.calculus.implementations;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.operations.AbstractOperation;
import neureka.calculus.Function;
import neureka.calculus.assembly.FunctionBuilder;

import java.util.ArrayList;
import java.util.List;

/**
 *  Instances of this implementation of the {@link Function} interface
 *  are leave nodes within the abstract syntax tree of a function, representing indexed inputs to a function.
 *  When parsing an expression into a function then these inputs are recognized by the character 'i' or 'I',
 *  followed by the character 'j' or 'J' (optinally wrapped by '[' and ']'), which is a placeholder for the index
 *  of the argument within the list/array of arguments passed to a concrete {@link Function} instance. <br>
 *  So for example, when creating a function by calling the following factory method...     <br>
 *                                                                                          <br>
 *  {@link Function#of}( "3 * sum( (I[j] + 4) * I[0] )" )                               <br>
 *                                                                                          <br>
 *  ...then the substrings "I[j]" will be parsed into instances of this class!              <br>
 *  The substring "I[0]" on the other hand will not be parsed into an instance of this class!
 */
public class FunctionVariable implements Function, GradientProvider {

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
    public boolean dependsOn( int index ) {
        return true;
    }

    @Override
    public Function getDerivative( int index ) {
        return Function.of( "1" );
    }

    @Override
    public List<Function> getSubFunctions() { return new ArrayList<>(); }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public Function newBuild( final String equation ) {
        if ( equation.contains( "g" ) ) _providesGradient = true;
        return this;
    }

    @Override
    public double call( final double[] inputs, int j ) {
        if ( j < 0 ) {
            double sum = 0;
            for ( int i = 0; i < inputs.length; i++ ) sum += call(inputs, i);
            return sum;
        }
        return inputs[j];
    }

    @Override
    public double derive( final double[] inputs, final int index ) {
        return 1.0;
    }

    @Override
    public double derive( double[] inputs, int index, int j ) {
        if (j != index) return 0;
        return derive(inputs, index);
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public Tsr<?> execute( Tsr<?>[] inputs, int j ) {
        return inputs[j];
    }

    @Override
    public Tsr<?> execute( Tsr<?>... inputs ) {
        StringBuilder exp = new StringBuilder("I[ 0 ]");
        for(int i=1; i<inputs.length; i++) exp.append("+I[").append(i).append("]");
        return new FunctionBuilder(Neureka.get().context()).build(exp.toString(), false).execute( inputs );
    }

    @Override
    public Tsr<?> executeDerive( Tsr<?>[] inputs, int index, int j ) {
        return (j != index) ? Tsr.of( inputs[ 0 ].shape(), 0.0 ) : executeDerive( inputs, index );
    }

    @Override
    public Tsr<?> executeDerive( Tsr<?>[] inputs, int index ) {
        return Tsr.of( inputs[ 0 ].shape(), 1.0 );
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public String toString() {
        return "I" + ( (this.providesGradient()) ? "g" : "" ) + "[j]";
    }


}
