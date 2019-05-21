package neureka.main.core.base;

import neureka.main.core.NVUtility;

public class NVData{

	protected NVNeuron[] Neuron;
	
	public NVData(NVNeuron[] newElement)
	{
		if(newElement==null){return;}
		this.Neuron = new NVNeuron[newElement.length];
		for(int Ei=0; Ei<newElement.length; Ei++){
			Neuron[Ei] = new NVNeuron(newElement[Ei].inputSize());
			Neuron[Ei].setActivation(newElement[Ei].getActivation());
			Neuron[Ei].setError(newElement[Ei].getError());
			Neuron[Ei].setOptimum(newElement[Ei].getOptimum());
			Neuron[Ei].setInputDerivative(newElement[Ei].getInputDerivative());
			Neuron[Ei].setRelationalDerivative(newElement[Ei].getRelationalDerivative());
			Neuron[Ei].setInput(newElement[Ei].getInput());
		}
	}
	
	public NVNeuron[] getDataAsElementArray() {return Neuron;}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public int size() {return Neuron.length;}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void load(NVData data)
	{
		if(data==null){return;}
		for(int Vi=0; Vi<data.size() && Vi<this.size(); Vi++)
		{//Todo: value ought to not have certain fields (Optimum, bias, weight, input!)
			Neuron[Vi].setActivation(data.getActivation(Vi));
			//Neuron[Vi].setError(value.getError(Vi));
			//Neuron[Vi].setOptimum(value.getOptimum(Vi));
			Neuron[Vi].setInputDerivative(data.getInputDerivative(Vi));
			Neuron[Vi].setRelationalDerivative(data.getRelationalDerivative(Vi));
			//Neuron[Vi].setOutput(value.Neuron[Vi].getOutput());
		}
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public boolean isDerivable()
	{
		for(int Vi = 0; Vi< Neuron.length; Vi++)
		{
			if(Neuron[Vi].hasError())
			{
				return  true;
			}
		}
		return false;
	}
	public void setDerivable(boolean derivable)
	{
		if(this.isDerivable()==derivable){return;}
		if(derivable)
		{
			for(int Di = 0; Di< Neuron.length; Di++)
			{
				Neuron[Di].setHasError(true);
				Neuron[Di].setHasActivation(true);
				Neuron[Di].setHasInput(true);
				Neuron[Di].setHasInputDerivative(true);
			}
		}
		else
		{
			for(int Di = 0; Di< Neuron.length; Di++)
			{
				Neuron[Di].setHasError(false);
			}
		}
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public synchronized double getError(int Vi) 
	{
		return Neuron[Vi].getError();
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public synchronized void setError(int Vi, double value) 
	{
		Neuron[Vi].setError(value);
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public synchronized void addToError(int Vi, double value) 
	{
		if(this.isDerivable()) 
		{
			Neuron[Vi].addToError(value);
		}
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public synchronized boolean isConvergent() 
	{
		if(isDerivable()==false) 
		{
			return false;
		}
		for(int Vi = 0; Vi< Neuron.length; Vi++)
		{
			if(Neuron[Vi].hasOptimum()){return true;}
		}
		return false;
	}
	public synchronized void setConvergence(boolean converge)
	{
		// Adds entry in Optimum in Neuron[i].
		// The node will converge towards this value (either by itself or indirectly (loss node)).
		// Last layer nodes converge with the Help of a loss node. Non last layer nodes converge without help!
		if(converge==this.isConvergent())
		{
			return;
		}
		if(converge)
		{
			if(this.isDerivable()==false)
			{
				setDerivable(true);
			}
			for(int Vi = 0; Vi< Neuron.length; Vi++)
			{
				Neuron[Vi].setHasOptimum(true);
			}
		}
		else
		{
			for(int Vi = 0; Vi< Neuron.length; Vi++)
			{
				Neuron[Vi].setHasOptimum(false);
			}
		}
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public synchronized double getOptimum(int Vi) 
	{
		if (Neuron[Vi].hasOptimum())
		{
			return Neuron[Vi].getOptimum();
		}    
		return 0;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public synchronized void setOptimum(int Vi, double value) 
	{
		if (Neuron[Vi].hasOptimum())
		{
			Neuron[Vi].setOptimum(value);
		}	    
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public synchronized void addToOptimum(int Vi, double value) 
	{
		if(this.isConvergent()) 
		{
			Neuron[Vi].calculateError();
		}	
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public double[] getActivation() 
	{
		double[] activation = new double[Neuron.length];
		for(int Di = 0; Di< Neuron.length; Di++)
		{
			activation[Di]= Neuron[Di].getActivation();
		}
		return activation;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public double getActivation(int Vi) 
	{
		if(Neuron.length<=Vi)
		{
			return 0;
		}
		return Neuron[Vi].getActivation();
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void setActivation(int Vi, double value) 
	{
		Neuron[Vi].setActivation(value);
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public synchronized double[][] getRelationalDerivative() {double[][] derivative = new double[Neuron.length][]; for(int i = 0; i< Neuron.length; i++) {derivative[i] = Neuron[i].getRelationalDerivative();}return derivative;}

	public synchronized double getRelationalDerivative(int Vi, int Ii) {if(Vi<0) {return 0;}return Neuron[Vi].getRelationalDerivative()[Ii];}

	public void setRelationalDerivative(double[][] derivative) {for(int Vi=0; Vi<this.size()&&Vi<derivative.length; Vi++) {
		Neuron[Vi].setRelationalDerivative(derivative[Vi]);}}

	public boolean hasRelationalDerivative(int Vi) {if(Neuron[Vi].getRelationalDerivative()!=null) {return true;}return false;}

	public double[] getRelationalDerivative(int Vi) {if(Vi<0) {return null;}return Neuron[Vi].getRelationalDerivative();}

	public void setRelationalDerivative(int Vi, double[] gradient) {
		Neuron[Vi].setRelationalDerivative(gradient);}

	public synchronized void setRelationalDerivative(int Vi, int Ii, double value) {
		Neuron[Vi].setRelationalDerivative(Ii, value);}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public double[][] getInputDerivative() {double[][] derivative = new double[Neuron.length][]; for(int i = 0; i< Neuron.length; i++) {derivative[i] = Neuron[i].getInputDerivative();}return derivative;}

	public double getInputDerivative(int Vi, int Ii) {if(Vi<0) {return 0;}return Neuron[Vi].getInputDerivative()[Ii];}

	public void setInputDerivative(double[][] derivative) {for(int Vi=0; Vi<this.size()&&Vi<derivative.length; Vi++) {
		Neuron[Vi].setInputDerivative(derivative[Vi]);}}

	public boolean hasInputDerivative(int Vi) {if(Neuron[Vi].getInputDerivative()!=null) {return true;}return false;}

	public double[] getInputDerivative(int Vi) {if(Vi<0) {return null;}return Neuron[Vi].getInputDerivative();}

	public void setInputDerivative(int Vi, double[] gradient) {
		Neuron[Vi].setInputDerivative(gradient);}

	public synchronized void setInputDerivative(int Vi, int Ii, double value) {
		Neuron[Vi].setInputDerivative(Ii, value);}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public synchronized double[][] getInput() {double[][] input = new double[Neuron.length][]; for(int i = 0; i< Neuron.length; i++) {input[i] = Neuron[i].getInput();}return input;}

	public synchronized double getInput(int Vi, int Ii) {if(Vi<0) {return 0;}return Neuron[Vi].getInput()[Ii];}

	public synchronized void setInput(double[][] input) {for(int i = 0; i< Neuron.length&&i<input.length; i++) {
		Neuron[i].setInput(input[i]);}}

	public boolean hasInput(int Vi) {if(Neuron[Vi].getInput()!=null) {return true;}return false;}

	public synchronized void setInput(int Vi, double[] input) {
		Neuron[Vi].setInput(input);}

	public synchronized void setInput(int Vi, int Ii, double value){
		Neuron[Vi].setInput(Ii, value);}

	public synchronized void setAllInput(int index, double input) {for(int i = 0; i< Neuron.length; i++) {
		Neuron[i].setInput(index, input);}}

	public synchronized void addToAllInput(int index, double input) {for(int i = 0; i< Neuron.length; i++) {
		Neuron[i].addToInput(index, input);}}

	public synchronized void addToInput(double[][] input) {for(int i = 0; i< Neuron.length&&i<input.length; i++) {for(int Ii = 0; Ii< Neuron[i].Input.length; Ii++) {
		Neuron[i].addToInput(Ii, input[i][Ii]);}}}

	public synchronized void addToInput(int Vi, int Ii, double value) {
		Neuron[Vi].addToInput(Ii, value);}

	public double[] getInput(int Vi) {if(Vi<0) {return null;}return Neuron[Vi].getInput();}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//@Override// NVNode
	public boolean 	  hasBiasGradient(int Vi, int Ii) {if(Vi==0) {if(Neuron[Vi].Input !=null) {if(Neuron[Vi].Input.length>Ii&& Neuron[Vi].Input[Ii].hasBiasGradient()) {return true;}}}return false;}
	//@Override// NVNode
	public double     getBiasGradient(int Vi, int Ii) {if(Vi==0) {if(Neuron[Vi].Input !=null) {if(Neuron[Vi].Input.length>Ii) {return Neuron[Vi].Input[Ii].BiasGradient;}}}return 0;}
	//@Override// NVNode
	public void	      setBiasGradient(int Vi, int Ii, double value) {if(Vi==0) {if(Neuron[Vi].Input !=null) {if(Neuron[Vi].Input.length>Ii) {
		Neuron[Vi].Input[Ii].BiasGradient=value;}}}}
	//@Override// NVNode
	public void	      addToBiasGradient(int Vi, int Ii, double value) {if(Vi==0) {if(Neuron[Vi].Input !=null) {if(Neuron[Vi].Input.length>Ii) {
		Neuron[Vi].Input[Ii].BiasGradient+=value;}}}}
	//-------------------------------------------------------------
	//@Override// NVNode
	public boolean 	  hasBiasGradient(int Ii) {if(Neuron[0].Input[Ii].hasBiasGradient()) {return true;}return false;}
	//@Override// NVNode
	public double[]   getBiasGradient(int Vi) {if(Vi>=0) {if(Neuron.length>Vi) {return Neuron[Vi].BiasGradient();}}return null;}
	//@Override// NVNode
	public void	      setBiasGradient(int Vi, double[] array) {if(Vi==0) {for(int i = 0; i< Neuron[Vi].Input.length&&i<array.length; i++) {
		Neuron[Vi].Input[i].BiasGradient=array[i];}}}
	//@Override// NVNode
	public void	      addToBiasGradient(int Vi, double[] array) {if(Vi==0) {for(int i = 0; i< Neuron[Vi].Input.length&&i<array.length; i++) {
		Neuron[Vi].Input[i].BiasGradient+=array[i];}}}
	//-------------------------------------------------------------
	//@Override// NVNode
	public boolean 	  hasBiasGradient() {for(int Ii=0; Ii<this.inputSize(); Ii++) {if(Neuron[0].hasBiasGradient(Ii)) {return true;}}return false;}
	//@Override// NVNode
	public double[][] getBiasGradient() {double[][] newBiasGradient = new double[Neuron.length][];for(int Di = 0; Di< Neuron.length; Di++) {newBiasGradient[Di]= Neuron[Di].BiasGradient();} return newBiasGradient;}
	//@Override// NVNode
	public void	      setBiasGradient(double[][] array) {if(array==null) {for(int Vi = 0; Vi< Neuron.length; Vi++) {for(int Ii = 0; Ii< Neuron[Vi].Input.length; Ii++) {
		Neuron[Vi].Input[Ii].BiasGradient=null;}}}else{for(int Vi = 0; Vi< Neuron.length&&Vi<array.length; Vi++) {this.setBiasGradient(Vi, array[Vi]);}}}
	//@Override// NVNode
	public void	      addToBiasGradient(double[][] array) {if(array!=null) {for(int Vi = 0; Vi< Neuron.length&&Vi<array.length; Vi++) {this.addToBiasGradient(Vi, array[Vi]);}}}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//@Override // NVNode
	public boolean 	  hasBias(int Vi, int Ii) {if(Vi==0) {if(Neuron[Vi].Input !=null) {if(Neuron[Vi].Input.length>Ii) {return true;}}}return false;}
	//@Override // NVNode
	public double     getBias(int Vi, int Ii) {if(Vi==0) {if(Neuron[Vi].Input !=null) {if(Neuron[Vi].Input.length>Ii) {return Neuron[Vi].Input[Ii].Bias;}}}return 0;}
	//@Override // NVNode
	public void	      setBias(int Vi, int Ii, double value) {if(Vi==0) {if(Neuron[Vi].Input !=null) {if(Neuron[Vi].Input.length>Ii) {
		Neuron[Vi].Input[Ii].Bias=value;}}}}
	//@Override // NVNode
	public void	      addToBias(int Vi, int Ii, double value) {if(Vi==0) {if(Neuron[Vi].Input !=null) {if(Neuron[Vi].Input.length>Ii) {
		Neuron[Vi].Input[Ii].Bias+=value;}}}}
	//-------------------------------------------------------------
	//@Override
	public boolean 	  hasBias(int Ii) {if(Neuron[0].Input[Ii].hasBias()) {return true;}return false;}
	//@Override // NVNode
	public double[]   getBias(int Vi) {if(Vi>=0) {if(Neuron.length>Vi) {return Neuron[Vi].Bias();}}return null;}
	//@Override // NVNode
	public void	      setBias(int Vi, double[] array) {if(Vi==0) {for(int i = 0; i< Neuron[Vi].Input.length&&i<array.length; i++) {
		Neuron[Vi].Input[i].Bias=array[i];}}}
	//@Override // NVNode
	public void	      addToBias(int Vi, double[] array) {if(Vi==0) {for(int i = 0; i< Neuron[Vi].Input.length&&i<array.length; i++) {
		Neuron[Vi].Input[i].Bias+=array[i];}}}
	//-------------------------------------------------------------
	//@Override // NVNode
	public boolean 	  hasBias() {for(int Ii=0; Ii<this.inputSize(); Ii++) {if(Neuron[0].hasBias(Ii)) {return true;}}return false;}
	//@Override // NVNode
	public double[][] getBias() {double[][] newBias = new double[Neuron.length][];for(int Di = 0; Di< Neuron.length; Di++) {newBias[Di]= Neuron[Di].Bias();} return newBias;}
	//@Override // NVNode
	public void	      setBias(double[][] array) {if(array==null) {for(int Vi = 0; Vi< Neuron.length; Vi++) {for(int Ii = 0; Ii< Neuron[Vi].Input.length; Ii++) {
		Neuron[Vi].Input[Ii].Bias=null;}}}else{for(int Vi = 0; Vi< Neuron.length&&Vi<array.length; Vi++) {this.setBias(Vi, array[Vi]);}}}
	//@Override // NVNode
	public void	      addToBias(double[][] array) {if(array!=null) {for(int Vi = 0; Vi< Neuron.length&&Vi<array.length; Vi++) {this.addToBias(Vi, array[Vi]);}}}
	//-------------------------------------------------------------
	//@Override// NVNode
	public boolean hasWeightGradient(int Vi, int Ii, int Ui) {if(hasWeightGradient(Vi,Ii)) {if(Neuron[Vi].Input[Ii].WeightGradient.length>Ui&&Ui>=0) {return true;}}return false;}
	//@Override// NVNode
	public double  getWeightGradient(int Vi, int Ii, int Ui) {if (hasWeightGradient(Vi, Ii, Ui)) {return Neuron[Vi].Input[Ii].WeightGradient[Ui];}return 0;}
	//@Override// NVNode
	public void    setWeightGradient(int Vi, int Ii, int Ui, double value) {if (hasWeightGradient(Vi, Ii, Ui)) {
		Neuron[Vi].Input[Ii].WeightGradient[Ui]=value;}}
	//@Override// NVNode
	public void    addToWeightGradient(int Vi, int Ii, int Ui, double value) {if (hasWeightGradient(Vi, Ii, Ui)) {
		Neuron[Vi].Input[Ii].WeightGradient[Ui]+=value;}}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//@Override // NVNode
	public boolean  hasWeightGradient(int Vi, int Ii) {if(hasWeightGradient(Vi)) {if(Neuron[Vi].Input.length>Ii&&Ii>=0) {if(Neuron[Vi].Input[Ii].WeightGradient!=null) {return true;}}}return false;}
	//@Override// NVNode
	public double[] getWeightGradient(int Vi, int Ii) {if(hasWeightGradient(Vi,Ii)) {return Neuron[Vi].Input[Ii].WeightGradient;}return null;}
	//@Override// NVNode
	public void     setWeightGradient(int Vi, int Ii, double[] array) {if(Neuron[Vi].Input[Ii]!=null) {
		Neuron[Vi].Input[Ii].WeightGradient=array;}}
	//@Override// NVNode
	public void     addToWeightGradient(int Vi, int Ii, double[] array) {if(Neuron[Vi].Input[Ii]!=null) {for(int i = 0; i< Neuron[Vi].Input[Ii].WeightGradient.length&&i<array.length; i++) {
		Neuron[Vi].Input[Ii].WeightGradient[i]=array[i];}}}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//@Override// NVNode
	public boolean    hasWeightGradient(int Vi) {if(Vi< Neuron.length) {if(Neuron[Vi].Input[0].WeightGradient!=null) {return true;}}return false;}
	//@Override// NVNode
	public double[][] getWeightGradient(int Vi) {return Neuron[Vi].WeightGradient();}
	//@Override// NVNode
	public void       setWeightGradient(int Vi, double[][] array) {if(array==null) {array = new double[this.inputSize()][];}for(int Ii = 0; Ii< Neuron[Vi].Input.length&&Ii<array.length; Ii++) {setWeightGradient(Vi,Ii,array[Ii]);}}
	//@Override// NVNode
	public void       addToWeightGradient(int Vi, double[][] array) {if(array==null) {array = new double[this.inputSize()][];}for(int i = 0; i< Neuron[Vi].Input.length&&i<array.length; i++) {addToWeightGradient(Vi,i,array[i]);}}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//@Override// NVNode
	public boolean 		hasWeightGradient() {if(Neuron[0].Input[0].WeightGradient!=null) {return true;}return false;}
	//@Override// NVNode
	public double[][][] getWeightGradient() {double[][][] array = new double[Neuron.length][][];for(int i = 0; i<array.length; i++) {array[i]= Neuron[i].WeightGradient();}return array;}
	//@Override// NVNode
	public void		    setWeightGradient(double[][][] array) {if(array==null) {array = new double[this.size()][][];}for(int Vi = 0; Vi<array.length&&Vi< Neuron.length; Vi++) {this.setWeightGradient(Vi, array[Vi]);}}
	//@Override// NVNode
	public void		    addToWeightGradient(double[][][] array) {if(array==null) {array = new double[this.size()][][];}for(int Vi = 0; Vi<array.length&&Vi< Neuron.length; Vi++) {this.addToWeightGradient(Vi, array[Vi]);}}
	//-------------------------------------------------------------
	//@Override// NVNode
	public boolean hasWeight(int Vi, int Ii, int Wi) {if(hasWeight(Vi,Ii)) {if(Neuron[Vi].Input[Ii].Weight.length>Wi&&Wi>=0) {return true;}}return false;}
	//@Override// NVNode
	public double  getWeight(int Vi, int Ii, int Wi) {return Neuron[Vi].Input[Ii].Weight[Wi];}
	//@Override// NVNode
	public void    setWeight(int Vi, int Ii, int Wi, double value)
	{
		if (Neuron[Vi]!=null&& Ii<Neuron[Vi].Input.length&& Neuron[Vi].Input[Ii]!=null)
		{
			if(Neuron[Vi].Input[Ii].Weight==null)
			{
				Neuron[Vi].Input[Ii].Weight=new double[Wi+1];
			}else if(Neuron[Vi].Input[Ii].Weight.length>=Wi){
				return;
			}
			Neuron[Vi].Input[Ii].Weight[Wi]=value;
		}
	}
	//@Override// NVNode
	public void    addToWeight(int Vi, int Ii, int Wi, double value) {if (hasWeight(Vi, Ii, Wi)) {
		Neuron[Vi].Input[Ii].Weight[Wi]+=value;}}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//@Override // NVNode
	public boolean  hasWeight(int Vi, int Ii)
	{if(Vi< Neuron.length&&Vi>=0)
	{if(Ii< Neuron[Vi].Input.length&&Ii>=0)
	{if(Neuron[Vi].Input[Ii].Weight!=null)
	{return true;}}}
	return false;}
	//@Override// NVNode
	public double[] getWeight(int Vi, int Ii) {if(hasWeight(Vi,Ii)) {return Neuron[Vi].Input[Ii].Weight;}return null;}
	//@Override// NVNode
	public void     setWeight(int Vi, int Ii, double[] array) {if(Neuron[Vi].Input[Ii]!=null) {
		Neuron[Vi].Input[Ii].Weight=array;}}
	//@Override// NVNode
	public void     addToWeight(int Vi, int Ii, double[] array) {if(Neuron[Vi].Input[Ii]!=null) {for(int i = 0; i< Neuron[Vi].Input[Ii].Weight.length&&i<array.length; i++) {
		Neuron[Vi].Input[Ii].Weight[i]=array[i];}}}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//@Override// NVNode
	public boolean    hasWeight(int Vi) {if(Vi< Neuron.length) {if(Neuron[Vi].Input[0].Weight!=null) {return true;}}return false;}
	//@Override// NVNode
	public double[][] getWeight(int Vi) {return Neuron[Vi].Weight();}
	//@Override// NVNode
	public void       setWeight(int Vi, double[][] array) {if(array==null) {array = new double[this.inputSize()][];}for(int Ii = 0; Ii< Neuron[Vi].Input.length&&Ii<array.length; Ii++) {setWeight(Vi,Ii,array[Ii]);}}
	//@Override// NVNode
	public void       addToWeight(int Vi, double[][] array) {if(array==null) {array = new double[this.inputSize()][];}for(int i = 0; i< Neuron[Vi].Input.length&&i<array.length; i++) {addToWeight(Vi,i,array[i]);}}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//@Override// NVNode
	public boolean 		hasWeight() {if(Neuron[0].Input[0].Weight!=null) {return true;}return false;}
	//@Override// NVNode
	public double[][][] getWeight() {double[][][] array = new double[Neuron.length][][];for(int i = 0; i<array.length; i++) {array[i]= Neuron[i].Weight();}return array;}
	//@Override// NVNode
	public void		    setWeight(double[][][] array) {if(array==null) {array = new double[this.size()][][];}for(int Vi = 0; Vi<array.length&&Vi< Neuron.length; Vi++) {this.setWeight(Vi, array[Vi]);}}
	//@Override// NVNode
	public void		    addToWeight(double[][][] array) {if(array==null) {array = new double[this.size()][][];}for(int Vi = 0; Vi<array.length&&Vi< Neuron.length; Vi++) {this.addToWeight(Vi, array[Vi]);}}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public void weightAll()
	{
		for(int Vi=0; Vi<this.size(); Vi++) {
			for(int Ii=0; Ii<this.inputSize(); Ii++) {
				if(Neuron[Vi].Input[Ii].hasWeight()==false) {
					int contextSize = 0;
					for(int Ni = 0; Ni< Neuron[Vi].Input[Ii].size(); Ni++) {contextSize++;}
					Neuron[Vi].Input[Ii].Weight = new double[contextSize];
				}
			}
		}
	}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public void unweightAll()
	{
		for(int Vi=0; Vi<this.size(); Vi++)
		{
			for(int Ii=0; Ii<this.inputSize(); Ii++)
			{
				Neuron[Vi].Input[Ii].Weight = null;
			}
		}
	}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public synchronized void freeInput()
	{
		for(int i = 0; i< Neuron.length; i++)
		{
			Neuron[i].setInput(null);
		}
	}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public void randomizeInput()
	{
		double[][] newInput = new double[this.size()][this.inputSize()];
		for(int Vi=0; Vi<this.size(); Vi++)
		{
			for(int Ii=0; Ii<this.inputSize(); Ii++)
			{
				newInput[Vi][Ii]= NVUtility.randomDouble();
			}
		}
		setInput(newInput);
	}
	//--------------------------------------
	public int inputSize() {
		return Neuron[0].Input.length;
	}
	//--------------------------------------
	public String toString(){
		String result = "";
		for(int Vi=0; Vi<this.size(); Vi++){
			result += toString(Vi);
		}
		return result;
	}
	public String toString(int Vi){
		String base =  "||   ";
		String result = "";
		String level = "     ";
		result += "[]=|D|=|A|=|T|=|A|==|F|=|I|=|E|=|L|=|D|=|S|==>\n"+base+"\n";
		result += base+"[Neuron]["+Vi+"]:\n"+base+"(\n";
		result += base+level+"[Activation]:("+this.getActivation(Vi)+"),\n";
		result += base+level+"[Error]:("+this.getError(Vi)+"),\n";
		result += base+level+"[Optimum]:("+this.getOptimum(Vi)+"),\n";
		for(int Ii = 0; Ii< Neuron[Vi].Input.length; Ii++){
			result += base+level+"[Value]["+Ii+"]:(\n";
			result += base+level+level+"[Value]:("+ Neuron[Vi].Input[Ii].Value +"),\n";
			result += base+level+level+"[Derivative]:("+ Neuron[Vi].Input[Ii].Derivative +"),\n";
			result += base+level+level+"[Bias]:("+ Neuron[Vi].Input[Ii].Bias+"),\n";
			result += base+level+level+"[BiasGradient]:("+ Neuron[Vi].Input[Ii].BiasGradient+"),\n";
			result += base+level+level+"[Weight]:(\n"+base+level+level+level;
			if(Neuron[Vi].Input[Ii].Weight!=null){
				for(int Wi = 0; Wi< Neuron[Vi].Input[Ii].Weight.length; Wi++){
					result += "["+Wi+"]:("+ Neuron[Vi].Input[Ii].Weight[Wi]+"), ";
				}
			}else{
				result+="null";
			}
			result += "\n"+base+level+level+"),\n";
			result += base+level+level+"[WeightGradient]:(\n"+base+level+level+level;
			if(Neuron[Vi].Input[Ii].WeightGradient!=null){
				for(int Wi = 0; Wi< Neuron[Vi].Input[Ii].Weight.length; Wi++){
					result += "["+Wi+"]:("+ Neuron[Vi].Input[Ii].WeightGradient[Wi]+"), ";
				}
			}else{
				result+="null";
			}
			result += "\n"+base+level+level+"),\n";
			result += base+level+"),\n";
		}
		result += base+");\n"+base+"\n";
		result += "[]=====================================================>>\n";
		return result;
	}

}
