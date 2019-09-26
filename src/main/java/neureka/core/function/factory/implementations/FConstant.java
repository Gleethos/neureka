package neureka.core.function.factory.implementations;

import neureka.core.T;
import neureka.core.function.IFunction;

public class FConstant implements IFunction
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
	public IFunction newBuild(String expression)
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
	public T activate(T[] inputs, int j) {
		return T.factory.newTensor(this._value, inputs[0].shape());
	}

	@Override
	public T activate(T[] inputs) {
		return T.factory.newTensor(this._value, inputs[0].shape());
	}

	@Override
	public T derive(T[] inputs, int index, int j) {
		return T.factory.newTensor(0, inputs[0].shape());
	}

	@Override
	public T derive(T[] inputs, int index) {
		return T.factory.newTensor(0, inputs[0].shape());
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}
