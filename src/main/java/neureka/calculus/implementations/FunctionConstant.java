package neureka.calculus.implementations;

import neureka.Tsr;
import neureka.backend.api.operations.AbstractOperation;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.args.Args;

import java.util.ArrayList;
import java.util.List;

/**
 *  Instances of this implementation of the {@link Function} interface
 *  are leave nodes within the abstract syntax tree of a function, representing constant numeric values to a function.
 *  When parsing an expression into a function then these constants are recognized by a series of digit characters
 *  optionally separated by '.' to represent decimal digits. <br>
 *  So for example, when creating a function by calling the following factory method...     	<br>
 *                                                                                          	<br>
 *  {@link Function#of}( "I[1] + (4 * I[0]) / 2.1" )                                      	<br>
 *                                                                                          	<br>
 *  ...then the substrings "4" and "2.1" will be parsed into instances of this class!   		<br>
 *
 */
public class FunctionConstant implements Function
{
	private final double _value;

	public double value() { return _value; }

	public FunctionConstant(String expression)
	{
		StringBuilder number = new StringBuilder();
		for ( int i = 0; i < expression.length(); i++ )
		{
			if (
					Character.isDigit(expression.charAt( i ))
							|| expression.charAt( i ) == '.'
							|| expression.charAt( i ) == '-'
							|| expression.charAt( i ) == '+'
			) {
				number.append( expression.charAt( i ) );
			}
		}
		_value = Double.parseDouble( number.toString() );
	}

	//------------------------------------------------------------------------------------------------------------------

	@Override
	public boolean isFlat() { return  true; }

	@Override
	public boolean isDoingAD() { return false; }

	@Override
	public AbstractOperation getOperation() { return null; }

	@Override
	public boolean dependsOn( int index ) { return false; }

	@Override
	public Function getDerivative( int index ) { return Function.of( "0" ); }

	@Override
	public List<Function> getSubFunctions() { return new ArrayList<>(); }

	//------------------------------------------------------------------------------------------------------------------

	@Override
    public double call( final double[] inputs, int j ) { return _value; }

	@Override
	public double derive( double[] inputs, int index ) { return 0; }

	@Override
	public double derive( double[] inputs, int index, int j ) { return 0; }

	//------------------------------------------------------------------------------------------------------------------

	@Override
	public Tsr<?> execute( Args arguments, Tsr<?>... tensors ) {
		if ( arguments.has(Arg.DerivIdx.class) && arguments.valOf(Arg.DerivIdx.class) >= 0 ) {
			return Tsr.of(
						tensors[ 0 ].getValueClass(),
						tensors[ 0 ].shape(),
						0.0
					)
					.getUnsafe()
					.setIsIntermediate( true );
		}
		return Tsr.of(
					tensors[ 0 ].getValueClass(),
					tensors[ 0 ].shape(),
					_value
				)
				.getUnsafe()
				.setIsIntermediate( true );
	}

	//------------------------------------------------------------------------------------------------------------------

	@Override
	public String toString() { return String.valueOf( _value ); }


}
