package neureka.main.core.base;

import neureka.main.core.NVUtility;

public class NVInput
{
	public double Value;
	public double Derivative;
	
	public Double   Bias=null;//  1 BIAS FOR EVERY INPUT
	public double[] Weight=null;// WEIGHTED CONNECTIONS TO OTHER NEURONS (Why 2 dimensions? -> First dimension INPUT SIZE, second -> connections

	public double[] WeightGradient=null;// WEIGHT GRADIENTS FOR WEIGHTS
	public Double   BiasGradient=null;// GRADIENTS FOR MULTIPLE (or 1) INPUT/S
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public NVInput(int size, boolean useBias)
	{
		if(size>0) {Weight = new double[size];}else {Weight = null;}
		if(useBias) 
		{
			Bias= new Double(0);
		}
	}
	//=============================================================================================================
	public int size() {if(Weight!=null) {return Weight.length;}return 0;}
	//---------------------------------------------------------------------------------------
	public boolean hasBias() {if(Bias!=null) {return true;}return false;}
	public boolean hasBiasGradient() {if(BiasGradient!=null) {return true;}return false;}
	//---------------------------------------------------------------------------------------
	public boolean hasWeight() {if(Weight!=null) {return true;}return false;}
	public boolean hasWeightGradient() {if(WeightGradient!=null) {return true;}return false;}
	//---------------------------------------------------------------------------------------
	public boolean addElementAt(int index, double value)
	{
		if(addAt(index)==false) {return false;}
		Weight[index]=value;
		return true;
	}
	//---------------------------------------------------------------------------------------
	public boolean addAt(int index)
	{
		return modifyAt(index, false);
	}
	public boolean removeAt(int index)
	{
		return modifyAt(index, true);
	}
	//---------------------------------------------------------------------------------------
	public boolean modifyAt(int index, boolean remove) 
	{
		NVUtility<Object> helper = new NVUtility<Object>();
		if(hasWeight()) 
		{
			if(Weight.length<=index) {return false;}
			Weight = helper.updateArray(Weight, index, remove);
		}
		if(hasWeightGradient()) 
		{
			WeightGradient = helper.updateArray(WeightGradient, index, remove);
		}
		return true;
	}
	//---------------------------------------------------------------------------------------

}
