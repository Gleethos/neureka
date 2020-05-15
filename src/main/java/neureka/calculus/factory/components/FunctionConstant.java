package neureka.calculus.factory.components;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.environment.OperationType;

public class FunctionConstant implements Function
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
	public boolean doesAD(){
		return false;
	}

	@Override
	public int id() {
		return -1;
	}

	@Override
	public OperationType type() {
		return null;
	}

	@Override
	public boolean dependsOn(int index){
		return false;
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
	public double call(double input){
		return activate(input);
	}
	@Override
	public double call(double[] inputs, int j){
		return inputs[j];
	}
	@Override
	public double call(double[] inputs){
		return activate(inputs);
	}

	@Override
	public Tsr call(Tsr input){
		return activate(input);
	}

	@Override
	public Tsr call(Tsr[] inputs, int j){
		return activate(inputs, j);
	}

	@Override
	public Tsr call(Tsr[] inputs) {
		return activate(inputs);
	}

	@Override
	public double activate(double input){
		return activate(new double[]{input});
	}

	@Override
    public double activate(final double[] inputs, int j) {
    	return _value;
    }

	@Override
	public double activate(double[] inputs) {
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

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	@Override
	public Tsr activate(Tsr input){
		return activate(new Tsr[]{input});
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

	@Override
	public String toString() {
		return _value +"";
	}


	@Override
	public ADAgent getADAgent(Tsr[] inputs, int i, boolean forward){
		return null;
	}


}
