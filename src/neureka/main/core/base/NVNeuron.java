package neureka.main.core.base;

import neureka.main.core.NVUtility;
import neureka.main.core.modul.calc.Function;

public class NVNeuron
{
	private final static byte 
	hasInput_MASK = 1,
	hasInputDerivative_MASK = 2,
	hasActivation_MASK = 4,
	hasError_MASK = 8,
	hasOptimum_MASK = 16,
	hasRelationalDerivative_MASK = 32;
	private byte flags = 0; // types: biasset, visited, convergent, derivable, dropout
	//--------------------------------
	public boolean hasInput() 
	{
		if((flags&hasInput_MASK)==hasInput_MASK) {return true;}
		return false;
	}
	public void setHasInput(boolean hasInput) {
		if(hasInput!=hasInput()) {
			if(hasInput) {flags+=hasInput_MASK;}
			else {flags-=hasInput_MASK;}
		}
	}
	//--------------------------------
	public boolean hasInputDerivative() {
		if((flags&hasInputDerivative_MASK)==hasInputDerivative_MASK) {return true;}
		return false;
	}
	public void setHasInputDerivative(boolean hasInputDerivative) {
		System.out.println(hasInputDerivative()+"; "+flags+", Set has input derivative: "+hasInputDerivative);
		if(hasInputDerivative!=hasInputDerivative()) {
			if(hasInputDerivative) {flags+=hasInputDerivative_MASK;}
			else {flags-=hasInputDerivative_MASK;}
		}
		System.out.println(flags+" Result: "+this.hasInputDerivative());
	}
	//--------------------------------
	public boolean hasActivation() {
		if((flags&hasActivation_MASK)==hasActivation_MASK) {return true;}
		return false;
	}
	public void setHasActivation(boolean hasActivation) {
		if(hasActivation!=hasInput()) {
			if(hasActivation) {flags+=hasActivation_MASK;}
			else {flags-=hasActivation_MASK;}
		}
	}
	//--------------------------------
	public boolean hasError() {
		if((flags&hasError_MASK)==hasError_MASK) {return true;}
		return false;
	}
	public void setHasError(boolean hasError) {
		if(hasError!=hasError()) {
			if(hasError) {flags+=hasError_MASK;}
			else {flags-=hasError_MASK;}
		}
	}
	//--------------------------------
	public boolean hasOptimum() {
		if((flags&hasOptimum_MASK)==hasOptimum_MASK) {return true;}
		return false;
	}
	public void setHasOptimum(boolean hasOptimum) {
		if(hasOptimum!=hasOptimum()) {
			if(hasOptimum) {flags+=hasOptimum_MASK;}
			else {flags-=hasOptimum_MASK;}
		}
	}
	//--------------------------------
	public boolean hasRelationalDerivative() 
	{
		if((flags&hasRelationalDerivative_MASK)==hasRelationalDerivative_MASK) {return true;}
		return false;
	}
	public void setHasRelationalDerivative(boolean hasRelationalDerivative) {
		if(hasRelationalDerivative!=hasRelationalDerivative()) {
			if(hasRelationalDerivative) {flags+=hasRelationalDerivative_MASK;}
			else {flags-=hasRelationalDerivative_MASK;}
		}
	}
	//--------------------------------
	private double Optimum;
	private double Activation;
	private double Error;
	private double[] RelationalDerivative;
	public NVInput[] Input;//WEIGHTS, BIASES AND THEIR GRADIENTS!
	//private double[][] Data;//STORAGE FOR ACTIVATIONS, DERIVATIVES AND CURRENT ERROR VALUE!
	//=============================================================================================================
	public int inputSize(){return Input.length;}
	public NVNeuron(int inputSize)
	{
		Input = new NVInput[Math.abs(inputSize)];
		for(int Ii = 0; Ii< Input.length; Ii++)
		{
			Input[Ii] = new NVInput(0, true);
		}
		this.setHasInput(true);
		this.setHasInputDerivative(true);
		this.setHasActivation(true);
		this.setHasError(true);
		this.setHasOptimum(false);
	}
	//=============================================================================================================
	//--------------------------------
	public double[][] getIO() 
	{
		double[][] IOData = new double[3][];
		
		IOData[0] = this.getOutput();
		IOData[1] = this.getInput();
		IOData[2] = this.getInputDerivative();
		return IOData;
	}
	public void setIO(double[][] newIO) 
	{
		if(newIO.length>=1) {
			setOutput(newIO[0]);
			if(newIO.length>=2) {
				setInput(newIO[1]);
				if(newIO.length>=3) {
					setInputDerivative(newIO[2]);
				}
			}
		}
	}
	//--------------------------------
	public double[] getOutput() 
	{
		double[] Output;
		if(this.hasActivation()) {
			if(this.hasError()) {
				if(this.hasOptimum()) {
					Output = new double[3];
					Output[2] = Optimum;
				}else {
					Output = new double[2];
				}
				Output[1] = Error;
			}else {
				Output = new double[1];
			}
			Output[0] = Activation;
		}else {
			Output = new double[0];
		}
		return Output;
	}
	public void setOutput(double[] newOutput) 
	{
		if(newOutput!=null && newOutput.length>=1) 
		{
			this.setHasActivation(true);
			Activation = newOutput[0];
			if(newOutput.length>=2) {
				this.setHasError(true);
				Error = newOutput[1];
				if(newOutput.length>=3) {
					this.setHasOptimum(true);
					Optimum = newOutput[2];
				}else {
					this.setHasOptimum(false);
				}
			}
			else {
				this.setHasError(false);
				this.setHasOptimum(false);
			}
		}else {
			this.setHasActivation(false);
			this.setHasError(false);
			this.setHasOptimum(false);
		}
	}
	//--------------------------------
	public double[] getInput() 
	{
		double[] Input = null;
		if(this.hasInput()) {
			Input = new double[this.Input.length];
			for(int Ii = 0; Ii< this.Input.length; Ii++) {
				Input[Ii] = this.Input[Ii].Value;
			}
		}
		return Input;
	}
	public void setInput(double[] newInput) 
	{
		if(newInput==null) 
		{
			for(int Ii = 0; Ii< Input.length; Ii++) {
				Input[Ii].Value
				= 0;
			}
			this.setHasInput(false);
			return;
		}
		this.setHasInput(true);
		System.out.println(Input.length+" ... "+newInput.length);
		for(int Ii = 0; Ii< Input.length && Ii<newInput.length; Ii++) {
			Input[Ii].Value
			= newInput[Ii];
		}
	}
	public void addToInput(double[] newInput) 
	{
		if(newInput==null) {return;}
		this.setHasInput(true);
		System.out.println(Input.length+" ... "+newInput.length);
		for(int Ii = 0; Ii< Input.length && Ii<newInput.length; Ii++) {
			Input[Ii].Value
			+= newInput[Ii];
		}
	}
	public void setInput(int Ii, double input) 
	{
		Input[Ii].Value = input;
	}
	public void addToInput(int Ii, double input) 
	{
		Input[Ii].Value += input;
	}
	//--------------------------------
	public double[] getInputDerivative() 
	{
		double[] InputDerivative = null;
		if(this.hasInputDerivative()) {
			InputDerivative = new double[this.Input.length];
			for(int Ii = 0; Ii< Input.length; Ii++) {
				InputDerivative[Ii] = Input[Ii].Derivative;
			}
		}
		
		return InputDerivative;
	}
	public double getInputDerivative(int Ii) 
	{
		if(this.hasInputDerivative()) {
			return Input[Ii].Derivative;
		}
		return 0.0;
	}
	public void setInputDerivative(double[] newDerivative) 
	{
		if(newDerivative==null) 
		{
			for(int Ii = 0; Ii< Input.length; Ii++) {
				Input[Ii].Derivative = 0;
			}
			this.setHasInputDerivative(false);
			return;
		}
		this.setHasInputDerivative(true);
		for(int Ii = 0; Ii< Input.length && Ii<newDerivative.length; Ii++) {
			Input[Ii].Derivative = newDerivative[Ii];
		}
	}
	public void addToInputDerivative(double[] newDerivative) 
	{
		if(newDerivative==null) {return;}
		this.setHasInputDerivative(true);
		for(int Ii = 0; Ii< Input.length && Ii<newDerivative.length; Ii++) {
			Input[Ii].Derivative += newDerivative[Ii];
		}
	}
	public void setInputDerivative(int Ii, double newDerivative) 
	{
		Input[Ii].Derivative =newDerivative;
	}
	public void addToInputDerivative(int Ii, double newDerivative) 
	{
		Input[Ii].Derivative +=newDerivative;
	}
	//--------------------------------
	public double[] getRelationalDerivative() {
		return RelationalDerivative;
	}
	public void setRelationalDerivative(double[] newRelationalDerivative) 
	{
		if(newRelationalDerivative==null) {this.setHasRelationalDerivative(false);}
		RelationalDerivative = newRelationalDerivative;
	}
	public void addToRelationalDerivative(double[] newRelationalDerivative) 
	{
		for(int Ri=0; Ri<newRelationalDerivative.length && Ri<RelationalDerivative.length; Ri++) 
		{
			RelationalDerivative[Ri] += newRelationalDerivative[Ri];
		}
		RelationalDerivative = newRelationalDerivative;
	}
	public void setRelationalDerivative(int Ri, double value) 
	{
		RelationalDerivative[Ri] = value;
	}
	public void addToRelationalDerivative(int Ri, double value) 
	{
		RelationalDerivative[Ri] += value;
	}
	//--------------------------------
	public boolean removeInputAt(int index) 
	{
		return modifyAt(index, true);
	}
	//--------------------------------
	public boolean addInputAt(int index)
	{
		return modifyAt(index, false);
	}
	//--------------------------------
	public boolean modifyAt(int index, boolean remove) 
	{
		if(Input.length<=index) {return false;}
		NVUtility<NVInput> helper = new NVUtility<NVInput>();
		Input = helper.updateArray(Input, index, remove, NVInput.class);
		if(remove==false) {
			
			Input[index] = new NVInput(0, true);
		}
		return true;
	}
	//---------------------------------------------------------------------------------------
	public double getActivation() 
	{
		return Activation;
	}
	public void setActivation(double value)
	{
		Activation=value;
	}
	public void addToActivation(double value)
	{
		Activation += value;
	}
	//--------------------------------
	public double getError() 
	{
		return Error;
	}
	public void setError(double value) 
	{
		Error=value;
	}
	public void addToError(double value)
	{
		Error+=value;
	}
	//--------------------------------
	public double getOptimum()
	{
		return Optimum;
	}
	public void setOptimum(double value) 
	{
		Optimum=value;
	}
	public void addToOptimum(double value)
	{
		Optimum+=value;
	}
	//---------------------------------------------------------------------------------------
	public double[] Bias() 
	{
		double[] bias = new double[Input.length];
		for(int i = 0; i< Input.length; i++)
		{
			if(Input[i]!=null)
			{
				if(Input[i].Bias!=null)
				{bias[i] = Input[i].Bias;}
				else 
				{bias[i] = 0.0;}
			}
			else
			{bias[i] = 0.0;}
		}
		return bias;
	}
	public double BiasGradient(int Ii) 
	{
		return Input[Ii].Bias;
	}
	public double[] BiasGradient() 
	{
		double[] biasgradient = new double[Input.length];
		for(int i=0; i<biasgradient.length; i++)
		{
			biasgradient[i] = Input[i].BiasGradient;
		}
		return biasgradient;
	}
	public double[] Weight(int Ii) 
	{
		return Input[Ii].Weight;
	}
	public double[][] Weight()
	{
		double[][] weight = new double[Input.length][];
		for(int Ii=0; Ii<weight.length; Ii++)
		{
			weight[Ii]= Input[Ii].Weight;
		}
		return weight;
	}
	public double[]   WeightGradient(int Ii) 
	{
		return Input[Ii].WeightGradient;
	}
	public double[][] WeightGradient()
	{
		double[][] weightgradient = new double[Input.length][];
		for(int Ii=0; Ii<weightgradient.length; Ii++)
		{
			weightgradient[Ii]= Input[Ii].WeightGradient;
		}
		return weightgradient;
	}
	public void setWeight(double[][] weight) 
	{
		for(int Ii = 0; Ii< Input.length&&Ii<weight.length; Ii++)
		{
			Input[Ii].Weight = weight[Ii];
		}
	}
	//---------------------------------------------------------------------------------------
	public void calculateError() 
	{
		double error;
		if(this.hasOptimum()) 
		{error = 2*(Optimum - Activation) / 50000;}//[0]:Activation; [1]:Error; [2]:Optimum; => Error = 2 * (Optimum - Activation) / X
		else 
		{error=0;}
		Error = error;
	}
	public void nullGradients() 
	{
		for(int Ii = 0; Ii< Input.length; Ii++)
		{
			Input[Ii].WeightGradient=null;
			Input[Ii].BiasGradient=null;
		}
	}
	public boolean hasWeight(int Ii) 
	{
		return Input[Ii].hasWeight();
	}
	public boolean hasBias(int Ii) 
	{
		return Input[Ii].hasBias();
	}
	public boolean hasWeightGradient(int Ii)
	{
		return Input[Ii].hasWeightGradient();
	}
	public boolean hasBiasGradient(int Ii) 
	{
		return Input[Ii].hasWeight();
	}
	public void activateOn(Function Function)
	{
		if(Function!=null) 
		{
			double[] Bias = new double[Input.length];
			for(int Ii = 0; Ii< Input.length; Ii++)
			{
				Bias[Ii]= Input[Ii].Bias;
			}
			double[] input = this.getInput();
			if(Bias!=null){
				for(int Ii = 0; Ii< input.length; Ii++)
				{
					input[Ii] += Bias[Ii];
				}
			}
			Activation = Function.activate(input);
			for(int Ii = 0; Ii< Input.length; Ii++)
			{
				Input[Ii].Derivative = Function.derive(input, Ii);
			}
		}
		else 
		{
			int size = Input.length;
			Activation=1;
			for(int Ii=0; Ii<size; Ii++) 
			{
				Activation*= Input[Ii].Value; //LocalData[1]*=LocalData[Ii+2];
			}
			for(int Ii=0; Ii<size; Ii++) 
		    {
				Input[Ii].Derivative =1;
				
				for(int i=0; i<size; i++) 
				{
					if(Ii!=i) {
						Input[Ii].Derivative *= Input[i].Value;}
				}
			}
		}
	}
	
}
