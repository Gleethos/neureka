package neureka.calculus.implementations;

import neureka.Tsr;
import neureka.backend.api.operations.AbstractOperation;
import neureka.calculus.Function;
import neureka.calculus.AbstractBaseFunction;

public class FunctionConstant extends AbstractBaseFunction
{
	private double _value;
	public double value() {
		return this._value;
	}

	//------------------------------------------------------------------------------------------------------------------

	@Override
	public boolean isFlat() {
		return  false;
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
		return false;
	}

	@Override
	public Function newBuild(String expression)
	{	
		String number = "";
		for(int i=0; i<expression.length(); i++) 
		{
			if (expression.charAt( i )>='0'
			|| expression.charAt( i )<='9'
			|| expression.charAt( i )=='.'
			|| expression.charAt( i )=='-'
			|| expression.charAt( i )=='+')
			{
				number += expression.charAt( i );
			}
		}
		_value = Double.parseDouble(number);
		return this;
	}

	//------------------------------------------------------------------------------------------------------------------

	@Override
    public double call(final double[] inputs, int j) {
    	return _value;
    }

	@Override
	public double call(double[] inputs) {
		return _value;
	}

	@Override
	public double derive(double[] inputs, int index) {
		return 0;
	}

	@Override
	public double derive(double[] inputs, int index, int j) {
		return 0;
	}

	//------------------------------------------------------------------------------------------------------------------

	@Override
	public Tsr call(Tsr[] inputs, int j) {
		return new Tsr(inputs[ 0 ].shape(), this._value);
	}

	@Override
	public Tsr call(Tsr[] inputs) {
		return new Tsr(inputs[ 0 ].shape(), this._value);
	}

	@Override
	public Tsr derive(Tsr[] inputs, int index, int j) {
		return new Tsr(inputs[ 0 ].shape(), 0.0);
	}

	@Override
	public Tsr derive(Tsr[] inputs, int index) {
		return new Tsr(inputs[ 0 ].shape(), 0.0);
	}

	//------------------------------------------------------------------------------------------------------------------

	@Override
	public String toString() {
		return _value +"";
	}


}
