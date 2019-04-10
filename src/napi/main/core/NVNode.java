package napi.main.core;

import napi.main.core.NVNode.NAction;
import napi.main.core.NVNode.NCondition;
import napi.main.exec.NVExecutable;

public abstract interface NVNode 
{
	public NVNode[] find(NCondition condition);
	
	public interface NCondition
	{
		public abstract boolean check(NVNode node); 
	}	
	public interface NAction
	{
		public void act(NVNode unit);
	}
	//---------------------------------------------------
    public abstract void apply(NAction Actor);
    public int size();
    //-------------------------------------------------------------  
    public static interface Typer {public boolean check(NVNode node);}
    public abstract boolean is(Typer type);
	//-------------------------------------------------------------
	public abstract NVNode   	asNode();
	//-------------------------------------------------------------
	public abstract boolean isExecutable();
	public abstract NVExecutable asExecutable();
	
	public abstract boolean isRoot();
	public abstract NVRoot   asRoot();
	
	public abstract boolean isNeuron();
	public abstract NVNeuron asNeuron();
	
	public abstract NVertex   asCore();
	//=============================================================
	//-------------------------------------------------------------
	public abstract boolean 	 hasBiasGradient(int Vi, int Ii);
	public abstract double      getBiasGradient(int Vi, int Ii);
	public abstract void	     setBiasGradient(int Vi, int Ii, double value);
	public abstract void	     addToBiasGradient(int Vi, int Ii, double value);
		
	public abstract boolean 	 hasBiasGradient(int Vi);
	public abstract double[]    getBiasGradient(int Vi);
	public abstract void	     setBiasGradient(int Vi, double[] array);
	public abstract void	     addToBiasGradient(int Vi, double[] array);
	
	public abstract boolean 	 hasBiasGradient();
	public abstract double[][]  getBiasGradient();
	public abstract void	     setBiasGradient(double[][] array);
	public abstract void	     addToBiasGradient(double[][] array);
	//-------------------------------------------------------------
	public abstract boolean 	 hasBias(int Vi, int Ii);
	public abstract double      getBias(int Vi, int Ii);
	public abstract void	     setBias(int Vi, int Ii, double value);
	public abstract void	     addToBias(int Vi, int Ii, double value);
	
	public abstract boolean 	 hasBias(int Vi);
	public abstract double[]    getBias(int Vi);
	public abstract void	     setBias(int Vi, double[] array);
	public abstract void	     addToBias(int Vi, double[] array);
	
	public abstract boolean 	 hasBias();
	public abstract double[][]  getBias();
	public abstract void	     setBias(double[][] array);
	public abstract void	     addToBias(double[][] array);
	//-------------------------------------------------------------
	public abstract boolean 	 hasWeightGradient(int Vi, int Ii, int Ui);
	public abstract double      getWeightGradient(int Vi, int Ii, int Ui);
	public abstract void	     setWeightGradient(int Vi, int Ii, int Ui, double value);
	public abstract void	     addToWeightGradient(int Vi, int Ii, int Ui, double value);
	
	public abstract boolean 	 hasWeightGradient(int Vi, int Ii);
	public abstract double[]    getWeightGradient(int Vi, int Ii);
	public abstract void	     setWeightGradient(int Vi, int Ii, double[] array);
	public abstract void	     addToWeightGradient(int Vi, int Ii, double[] array);
	
	public abstract boolean 	 hasWeightGradient(int Vi);
	public abstract double[][]  getWeightGradient(int Vi);
	public abstract void	     setWeightGradient(int Vi, double[][] array);
	public abstract void	     addToWeightGradient(int Vi, double[][] array);
	
	public abstract boolean 	 hasWeightGradient();
	public abstract double[][][]getWeightGradient();
	public abstract void        setWeightGradient(double[][][] array);
	public abstract void        addToWeightGradient(double[][][] array);
	//-------------------------------------------------------------
	public abstract boolean 	 hasWeight(int Vi, int Ii, int Ui);
	public abstract double      getWeight(int Vi, int Ii, int Ui);
	public abstract void	     setWeight(int Vi, int Ii, int Ui, double value);
	public abstract void	     addToWeight(int Vi, int Ii, int Ui, double value);
	
	public abstract boolean 	 hasWeight(int Vi, int Ii);
	public abstract double[]    getWeight(int Vi, int Ii);
	public abstract void	     setWeight(int Vi, int Ii, double[] array);
	public abstract void	     addToWeight(int Vi, int Ii, double[] array);
	
	public abstract boolean 	 hasWeight(int Vi);
	public abstract double[][]  getWeight(int Vi);
	public abstract void	     setWeight(int Vi, double[][] array);
	public abstract void	     addToWeight(int Vi, double[][] array);
	
	public abstract boolean 	 hasWeight();
	public abstract double[][][]getWeight();
	public abstract void        setWeight(double[][][] array);
	public abstract void        addToWeight(double[][][] array);
	//-------------------------------------------------------------
	public abstract void removeInput(int targIi); 
	public abstract void addInput(int targIi);
	public abstract void addInput(int targIi, int Fid); 
	//-------------------------------------------------------------
	public abstract void swapConnection(NVNode former, NVNode replacement); 
	//-------------------------------------------------------------
	public abstract void disconnect(NVNode   node); 
	public abstract void disconnect(NVNode[] array, int targIi); 
	public abstract void disconnect(NVNode   node,  int targIi);
	//-------------------------------------------------------------
	public abstract void connect(NVNode node); 
	public abstract void connect(NVNode[] array, int targIi); 
	public abstract void connect(NVNode node, int targIi);
	//-------------------------------------------------------------
	public abstract void  copy(NVertex newCore);
	//-------------------------------------------------------------
	public abstract double   getActivation(int Vi);
	public abstract long 	 getActivitySignal();
	//--------------------------------------
    public abstract boolean  isDerivable();
    //-------------------------------------
	public abstract void     addToError(int Vi, double value);
	public abstract double   getError(int Vi);
	public abstract void     setError(int Vi, double value);
	//--------------------------------------
	public abstract boolean  isConvergent();
	//--------------------------------------
	public abstract void     addToOptimum(int Vi, double value);
	public abstract double   getOptimum(int Vi);
	public abstract void     setOptimum(int Vi, double value);
	//=============================================================

	double[] getActivation();
	
}
