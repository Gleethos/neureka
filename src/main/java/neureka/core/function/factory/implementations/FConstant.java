package neureka.core.function.factory.implementations;

import neureka.core.Tsr;
import neureka.core.function.Function;

public class FConstant implements Function
{
	private double _value;
	public double value(){
		return this._value;
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override
	public boolean isFlat(){
		return  false;
	}

	@Override
	public int id() {
		return -1;
	}

	@Override
	public String type() {
		return "_value";
	}


	@Override
	public Function newBuild(String expression)
	{	
		String number = "";
		for(int i=0; i<expression.length(); i++) 
		{
			if(expression.charAt(i)>='0' 
			|| expression.charAt(i)<='9' 
			|| expression.charAt(i)=='.' 
			|| expression.charAt(i)=='-' 
			|| expression.charAt(i)=='+') 
			{
				number += expression.charAt(i);
			}
		}
		_value = Double.parseDouble(number);
		return this;
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override
    public double activate(final double[] inputs, int j) {
    	return _value;
    }
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override
	public double activate(double[] inputs) {
		return _value;
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override
	public double derive(double[] inputs, int index) {
		return 0;
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override
	public double derive(double[] inputs, int index, int j) {
		return 0;
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override
	public String toString() {
		return _value +"";
	}

	@Override
	public Tsr activate(Tsr[] inputs, int j) {
		return new Tsr(inputs[0].shape(), this._value);
	}

	@Override
	public Tsr activate(Tsr[] inputs) {
		return new Tsr(inputs[0].shape(), this._value);
	}

	@Override
	public Tsr derive(Tsr[] inputs, int index, int j) {
		return new Tsr(inputs[0].shape(), 0.0);
	}

	@Override
	public Tsr derive(Tsr[] inputs, int index) {
		return new Tsr(inputs[0].shape(), 0.0);
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}
