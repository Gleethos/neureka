package neureka.calculus.implementations;

import neureka.Tsr;
import neureka.backend.api.operations.AbstractOperation;
import neureka.calculus.Function;

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
	private double _value;

	public double value() { return this._value; }

	//------------------------------------------------------------------------------------------------------------------

	@Override
	public boolean isFlat() { return  false; }

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

	@Override
	public Function newBuild( String expression )
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
		return this;
	}

	//------------------------------------------------------------------------------------------------------------------

	@Override
    public double call( final double[] inputs, int j ) { return _value; }

	@Override
	public double derive( double[] inputs, int index ) { return 0; }

	@Override
	public double derive( double[] inputs, int index, int j ) { return 0; }

	//------------------------------------------------------------------------------------------------------------------

	@Override
	public Tsr<?> execute( Tsr<?>[] inputs, int j ) { return Tsr.of( inputs[ 0 ].shape(), this._value ); }

	@Override
	public Tsr<?> executeDerive( Tsr<?>[] inputs, int index, int j ) { return Tsr.of( inputs[ 0 ].shape(), 0.0 ); }

	//------------------------------------------------------------------------------------------------------------------

	@Override
	public String toString() { return String.valueOf( _value ); }


}
