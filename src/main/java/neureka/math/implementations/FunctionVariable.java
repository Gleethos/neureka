package neureka.math.implementations;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.math.args.Args;
import neureka.math.parsing.FunctionParser;

import java.util.ArrayList;
import java.util.List;

/**
 *  Instances of this implementation of the {@link Function} interface
 *  are leave nodes within the abstract syntax tree of a function, representing indexed inputs to a function.
 *  When parsing an expression into a function then these inputs are recognized by the character 'i' or 'I',
 *  followed by the character 'j' or 'J' (optionally wrapped by '[' and ']'), which is a placeholder for the index
 *  of the argument within the list/array of arguments passed to a concrete {@link Function} instance. <br>
 *  So for example, when creating a function by calling the following factory method...     <br>
 *                                                                                          <br>
 *  {@link Function#of}( "3 * sum( (I[j] + 4) * I[0] )" )                               <br>
 *                                                                                          <br>
 *  ...then the substrings "I[j]" will be parsed into instances of this class!              <br>
 *  The substring "I[0]" on the other hand will not be parsed into an instance of this class!
 */
public final class FunctionVariable implements Function, GradientProvider
{
    private final boolean _providesGradient;


    public FunctionVariable( String equation ) { _providesGradient = equation.contains("g"); }

    @Override public boolean providesGradient() { return _providesGradient; }

    @Override public boolean isFlat() { return true; }

    @Override public boolean isDoingAD() { return false; }

    @Override public AbstractOperation getOperation() { return null; }

    @Override public boolean dependsOn( int index ) { return true; }

    @Override public Function getDerivative( int index ) { return Function.of( "1" ); }

    @Override public List<Function> getSubFunctions() { return new ArrayList<>(); }

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
    public double derive( final double[] inputs, final int index ) { return 1.0; }

    @Override
    public double derive( double[] inputs, int index, int j ) {
        if ( j != index ) return 0;
        return derive( inputs, index );
    }

    @Override
    public Tsr<?> execute( Args arguments, Tsr<?>... inputs ) {
        int d = ( arguments.has(Arg.DerivIdx.class) ? arguments.valOf(Arg.DerivIdx.class) : -1 );
        int j = ( arguments.has(Arg.VarIdx.class)   ? arguments.valOf(Arg.VarIdx.class)   : -1 );
        if ( d >= 0 ) {
            if ( j < 0 )
                return Tsr.of( inputs[ 0 ].shape(), 1.0 ).getMut().setIsIntermediate( true );

            return j != d ? Tsr.of( inputs[ 0 ].shape(), 0.0 ).getMut().setIsIntermediate( true ) : executeDerive(inputs, d );
        }
        if ( j < 0 ) {
            StringBuilder exp = new StringBuilder("I[ 0 ]");

            for (int i = 1; i < inputs.length; i++ )
                exp.append("+I[").append(i).append("]");

            return new FunctionParser( Neureka.get().backend() )
                                        .parse(exp.toString(), false)
                                        .execute(inputs);
        }
        return inputs[j];
    }

    @Override
    public String toString() { return "I" + ( (this.providesGradient()) ? "g" : "" ) + "[j]"; }


}
