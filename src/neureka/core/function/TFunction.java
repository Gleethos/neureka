
package neureka.core.function;

import neureka.core.T;

public interface TFunction
{
	public TFunction newBuild(String expression);
	public String toString();
	public boolean isFlat();
	public int id();
	//-------------------------------------------
	public double activate(double[] input, int j);// Iteration over input via j !
	public double activate(double[] input);
	//-----------------------------------------------------
	public double derive(double[] input, int index, int j);
	public double derive(double[] input, int index);
	//--------------------------------------------------------------
	//-------------------------------------------
	public T activate(T[] input, int j);// Iteration over input via j !
	public T activate(T[] input);
	//-----------------------------------------------------
	public T derive(T[] input, int index, int j);
	public T derive(T[] input, int index);
	//--------------------------------------------------------------
}

 