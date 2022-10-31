package neureka.calculus.implementations;


import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.args.Args;
import neureka.calculus.assembly.FunctionParser;

import java.util.ArrayList;
import java.util.List;

/**
 *  Instances of this implementation of the {@link Function} interface
 *  are leave nodes within the abstract syntax tree of a function, representing inputs to a function.
 *  When parsing an expression into a function then these inputs are recognized by the character 'i' or 'I',
 *  followed by a whole number starting at zero (optionally wrapped by '[' and ']'), which is the index
 *  of the argument within the list/array of arguments passed to a concrete {@link Function} instance. <br>
 *  So for example, when creating a function by calling the following factory method...     <br>
 *                                                                                          <br>
 *  {@link Function#of}( "I[1] + (4 * I[0]) / 2" )                                      <br>
 *                                                                                          <br>
 *  ...then the substrings "I[1]" and "I[0]" will be parsed into instances of this class!   <br>
 *  When calling this function by passing two arguments, let's say (first, second) then
 *  the {@link FunctionInput} "I[0]" will pick the first argument, whereas "I[1]"
 *  will pick the second argument when evaluating the array of arguments.
 *
 */
public class FunctionInput implements Function, GradientProvider
{
    private final int _index;


    public static Function of(String equation) {
        if ( equation.charAt( 0 ) == '-' )
            return new FunctionParser( Neureka.get().backend() )
                    .parse(
                            equation.substring(1)+"*-1",
                            true
                    ); // TODO: This might be false!
        int number = 0;
        for ( int i = 0; i < equation.length(); ++i) {
            if ( equation.charAt( i ) <= '9' && equation.charAt( i ) >= '0' ) {
                number *= 10;
                number += Integer.parseInt(equation.charAt( i ) + "");
            }
        }
        if ( equation.contains("g") ) {
            number = -( number + 1 );
        }
        return new FunctionInput(number);
    }


    private FunctionInput( int number ) { _index = number; }


    public int index() { return ( this.providesGradient() ? ( Math.abs(_index) - 1 ) : _index ); }

    @Override public boolean providesGradient() { return ( _index < 0 ); }

    @Override public boolean isFlat() { return true; }

    @Override public boolean isDoingAD() { return false; }

    @Override public AbstractOperation getOperation() { return null; }

    @Override public boolean dependsOn( int index ) { return index() == index; }

    @Override public Function getDerivative( int index ) { return ( index == _index ) ? Function.of( "1" ) : Function.of( "0" ); }

    @Override public List<Function> getSubFunctions() { return new ArrayList<>(); }

    private Tsr<?> _extract( Tsr<?> t )
    {
        if ( this.providesGradient() ) {
            Tsr<?> gradient = t.gradient().orElse(null);
            if ( t.rqsGradient() ) {
                if ( gradient == null ) {
                    gradient = Tsr.of( t.getItemType(), t.shape(), 0.0 );
                    t.set( (Tsr) gradient );
                }
                return gradient;
            }
            throw new IllegalArgumentException(
                    "The provided tensor does not require gradients, this function input however " +
                    "expects to receive such tensors (gradient receivers)."
            );
        }
        return t;
    }

    @Override
    public double call( final double[] inputs, int j ) {
        if ( j < 0 ) {
            return inputs[ ( _index >= 0 ) ? _index : ( Math.abs( _index ) - 1 ) ];
        }
        return inputs[index()];
    }

    @Override public double derive( final double[] inputs, final int index ) { return ( index == index() ) ? 1 : 0; }

    @Override
    public double derive( double[] inputs, int index, int j ) {
        if ( j < 0 || j == index() )
            return derive( inputs, index );
        else
            return 0;
    }

    @Override
    public Tsr<?> execute( Args arguments, Tsr<?>... inputs ) {
        int d = ( arguments.has(Arg.DerivIdx.class) ? arguments.valOf(Arg.DerivIdx.class) : -1 );
        if ( d >= 0 )
            return ( d == index() )
                ? Tsr.of( inputs[ 0 ].getItemType(), inputs[ 0 ].shape(), 1.0 ).getMut().setIsIntermediate( true )
                : Tsr.of( inputs[ 0 ].getItemType(), inputs[ 0 ].shape(), 0.0 ).getMut().setIsIntermediate( true );

        if ( index() >= inputs.length )
            throw new IllegalArgumentException(
                "Function input '"+index()+"' not satisfied! " +
                "Please supply at least "+(index()+1)+" input tensors."
            );

        return _extract( inputs[ index() ] );
    }

    @Override
    public String toString() { return "I" + ( this.providesGradient() ? "g" : "" ) + "[" + index() + "]"; }

}
