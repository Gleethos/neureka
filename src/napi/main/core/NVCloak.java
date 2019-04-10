package napi.main.core;


import java.math.BigInteger;
import java.util.ArrayList;

import napi.main.exec.NVExecutable;
import ngui.NPanelNode;

public class NVCloak implements NVNode{

	private NVertex 	 Host;
	//---------------------------------------
	private double[][] PublicData;
	private long 	   PublicSignal;
	//---------------------------------------
	private boolean abandonYourself = false;
	//---------------------------------------
	
	//CONSTRUCTOR:
	public NVCloak(NVertex neuron){Host = neuron; }
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void abandonYourself() {abandonYourself=true;}
	//-----------------------------------------------------------
	public void       setSignal(long signal) 		 {PublicSignal = signal;}
	public void       setPublicData(double[][] Data) {PublicData = Data;}
	public double[][] getPublicData()                {return PublicData;}
	//-----------------------------------------------------------
	
	//NNode implemntation:
	//====================================================================================================
	
	@Override
	public boolean is(NVertex.Typer type) {return type.check(Host);}
	
	@Override
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public NVNode asNode() {
		if(abandonYourself) {return Host.asNode();}
		return this;
	}
	//-----------------------------------------------------------------------

	//GETTING UNDERLYING COMPONENTS OF NNODE
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override
	public void                 copy(NVertex newNeuron) {Host.copy(newNeuron);}
	@Override
	public synchronized NVertex asCore(){return Host;}

	@Override
	public boolean hasBiasGradient(int Vi, int Ii) {
		return false;
	}

	@Override
	public double getBiasGradient(int Vi, int Ii) {
		return 0;
	}

	@Override
	public void setBiasGradient(int Vi, int Ii, double value) {

	}

	@Override
	public void addToBiasGradient(int Vi, int Ii, double value) {

	}

	@Override
	public boolean hasBiasGradient(int Vi) {
		return false;
	}

	@Override
	public double[] getBiasGradient(int Vi) {
		return new double[0];
	}

	@Override
	public void setBiasGradient(int Vi, double[] array) {

	}

	@Override
	public void addToBiasGradient(int Vi, double[] array) {

	}

	@Override
	public boolean hasBiasGradient() {
		return false;
	}

	@Override
	public double[][] getBiasGradient() {
		return new double[0][];
	}

	@Override
	public void setBiasGradient(double[][] array) {

	}

	@Override
	public void addToBiasGradient(double[][] array) {

	}

	//-----------------------------------------------------------------------
	public void setCore(NVertex newCore) 
	{
		Host = newCore;
	}
	//-----------------------------------------------------------------------
	@Override
	public boolean isDerivable() 
	{
		if(PublicData!=null) 
		{
			if(PublicData.length>1) 
			{
				return true;
			}
		}
		return Host.isDerivable();
	}
	//ERROR / OPTIMUM HANDLING:
	@Override
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void setError(int Ui, double value) 
	{
		if(PublicData != null){PublicData[Ui][1] = value;}
	}
	//Accessed by perceiving neurons (output neurons): 
	@Override
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	synchronized public void addToError(int Ui, double value)
	{
		if(PublicData!=null)
		{
			PublicData[Ui][1] += value;
		}
		else
		{
			System.out.println("PublicData is null!");
		}
	}
	@Override
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	synchronized public double getError(int Ui)
	{
	if(PublicData != null) {return PublicData[Ui][1];}
	return 0;
	}
	//-----------------------------------------------------------------------
	@Override
	synchronized public boolean isConvergent() 
	{
		if(PublicData!=null) 
		{
			if(PublicData[0].length>2) {return true;}
		}
		return Host.isConvergent();
	}
	@Override
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	synchronized public void addToOptimum(int Ui, double value) {
		if(isConvergent()) 
		{
			PublicData[Ui][2]+=value;
		}
	}
	//-----------------------------------------------------------------------

	@Override
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	synchronized public double getOptimum(int Ui) 
	{
		if(PublicData!=null) 
		{
			if(PublicData[Ui].length>=3) {return PublicData[Ui][2];	}
		}
		else {
			System.out.println("public data null :E");
		}
		return 0;
	}
	//-----------------------------------------------------------------------

	@Override
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	synchronized public void setOptimum(int Ui, double value) 
	{
		if(PublicData != null) 
		{
			if(PublicData[0].length>2) {PublicData[Ui][2]=value;}
		}
		return;
	}

	@Override
	public double[] getActivation() {
		return new double[0];
	}
	//-----------------------------------------------------------------------

	@Override
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	synchronized public double getActivation(int Vi) 
	{
		if(PublicData!=null) 
		{
			return PublicData[Vi][0];
		}
		else 
		{
			return Host.getActivation(Vi);	
		}
	}
	@Override
	synchronized public long getActivitySignal() 
	{
		return PublicSignal;
	}
	//-----------------------------------------------------------------------
	@Override
	public void apply(NAction Actor) {Actor.act(Host);}
	//-----------------------------------------------------------------------
	@Override
	public NVNeuron asNeuron() {if(Host instanceof NVNeuron) {return (NVNeuron)this.Host;}return null;}
	@Override
	public NVRoot   asRoot() {if(Host instanceof NVRoot) {return (NVRoot)this.Host;}return null;}

	@Override
	public boolean isNeuron() {
		return false;
	}

	@Override
	public NVNode[] find(NCondition condition) {return Host.find(condition);}
	@Override
	public NVExecutable asExecutable() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean isRoot() {
		return false;
	}

	@Override
	public boolean isExecutable(){
		return this.Host.isExecutable();
	}
		//##############################################################################################################
	//==============================================================================================================

	@Override
	public int size() {
		// TODO Auto-generated method stub
		if(PublicData==null) {return this.Host.size();}
		return PublicData.length;
	}
	@Override
	public boolean hasBias(int Vi, int Ii) {return false;}
	@Override
	public double getBias(int Vi, int Ii) {return 0;}
	@Override
	public void setBias(int Vi, int Ii, double value) {}
	@Override
	public void addToBias(int Vi, int Ii, double value) {}
	@Override
	public boolean hasBias(int Vi) {return false;}
	@Override
	public double[] getBias(int Vi) {return null;}
	@Override
	public void setBias(int Vi, double[] array) {}
	@Override
	public void addToBias(int Vi, double[] array) {}
	@Override
	public boolean hasBias() {return false;}
	@Override
	public double[][] getBias() {return null;}
	@Override
	public void setBias(double[][] array) {}
	@Override
	public void addToBias(double[][] array) {}

	@Override
	public boolean hasWeightGradient(int Vi, int Ii, int Ui) {
		return false;
	}

	@Override
	public double getWeightGradient(int Vi, int Ii, int Ui) {
		return 0;
	}

	@Override
	public void setWeightGradient(int Vi, int Ii, int Ui, double value) {

	}

	@Override
	public void addToWeightGradient(int Vi, int Ii, int Ui, double value) {

	}

	@Override
	public boolean hasWeightGradient(int Vi, int Ii) {
		return false;
	}

	@Override
	public double[] getWeightGradient(int Vi, int Ii) {
		return new double[0];
	}

	@Override
	public void setWeightGradient(int Vi, int Ii, double[] array) {

	}

	@Override
	public void addToWeightGradient(int Vi, int Ii, double[] array) {

	}

	@Override
	public boolean hasWeightGradient(int Vi) {
		return false;
	}

	@Override
	public double[][] getWeightGradient(int Vi) {
		return new double[0][];
	}

	@Override
	public void setWeightGradient(int Vi, double[][] array) {

	}

	@Override
	public void addToWeightGradient(int Vi, double[][] array) {

	}

	@Override
	public boolean hasWeightGradient() {
		return false;
	}

	@Override
	public double[][][] getWeightGradient() {
		return new double[0][][];
	}

	@Override
	public void setWeightGradient(double[][][] array) {

	}

	@Override
	public void addToWeightGradient(double[][][] array) {

	}

	@Override
	public boolean hasWeight(int Vi, int Ii, int Ui) {return false;}
	@Override
	public double getWeight(int Vi, int Ii, int Ui) {return 0;}
	@Override
	public void setWeight(int Vi, int Ii, int Ui, double value) {}
	@Override
	public void addToWeight(int Vi, int Ii, int Ui, double value) {}
	@Override
	public boolean hasWeight(int Vi, int Ii) {return false;}
	@Override
	public double[] getWeight(int Vi, int Ii) {return null;}
	@Override
	public void setWeight(int Vi, int Ii, double[] array) {}
	@Override
	public void addToWeight(int Vi, int Ii, double[] array) {}
	@Override
	public boolean hasWeight(int Vi) {return false;}
	@Override
	public double[][] getWeight(int Vi) {return null;}
	@Override
	public void setWeight(int Vi, double[][] array) {}
	@Override
	public void addToWeight(int Vi, double[][] array) {}
	@Override
	public boolean hasWeight() {return false;}
	@Override
	public double[][][] getWeight() {return null;}
	@Override
	public void setWeight(double[][][] array) {}
	@Override
	public void addToWeight(double[][][] array) {}
	@Override
	public void removeInput(int targIi) {}
	@Override
	public void addInput(int targIi) {}
	@Override
	public void addInput(int targIi, int Fid) {}
	@Override
	public void swapConnection(NVNode former, NVNode replacement) {}
	@Override
	public void disconnect(NVNode unit) {}
	@Override
	public void disconnect(NVNode[] array, int targIi) {}
	@Override
	public void disconnect(NVNode uni, int targIi) {}
	@Override
	public void connect(NVNode uni) {}
	@Override
	public void connect(NVNode[] array, int targIi) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void connect(NVNode node, int targIi) {

	}
}
