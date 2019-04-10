package napi.main.core.comp;

import napi.main.core.NVUtility;

public class NVData {

	protected NVElement[] Element;
	
	public NVData(NVElement[] newElement)
	{
		this.Element = newElement;
	}
	
	public NVElement[] getDataAsElementArray() {return Element;}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public int size() {return Element.length;}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void consume(NVData data)
	{
		for(int Vi=0; Vi<data.size() && Vi<this.size(); Vi++)
		{
			Element[Vi].setActivation(Element[Vi].getActivation());
			Element[Vi].setError(Element[Vi].getError());
			Element[Vi].setOptimum(Element[Vi].getOptimum());
			//Element[Vi].setOutput(data.Element[Vi].getOutput());
		}
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public boolean isDerivable() 
	{
		if(Element[0].getOutput().length>=2) 
		{
			return true;
		}
		return false;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public synchronized double getError(int Vi) 
	{
		return Element[Vi].getError();
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public synchronized void setError(int Vi, double value) 
	{
		Element[Vi].setError(value);
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public synchronized void addToError(int Vi, double value) 
	{
		if(this.isDerivable()) 
		{
			Element[Vi].getOutput()[1]+=value;
		}
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public synchronized boolean isConvergent() 
	{
		if(isDerivable()==false) 
		{
			return false;
		}
		if(Element[0].getOutput().length >= 3) 
		{
			return true;
		}
		return false;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public synchronized double getOptimum(int Vi) 
	{
		if (this.isConvergent()) 
		{
			if(Element[Vi].getOutput().length==3) 
			{
				return Element[Vi].getOutput()[2];
			}
		}    
		return 0;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public synchronized void setOptimum(int Vi, double value) 
	{
		if (this.isConvergent()) 
		{
			Element[Vi].setOptimum(value);
		}	    
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public synchronized void addToOptimum(int Vi, double value) 
	{
		if(this.isConvergent()) 
		{
			Element[Vi].calculateError();
		}	
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public double[] getActivation() 
	{
		double[] activation = new double[Element.length];
		for(int Di=0; Di<Element.length; Di++) 
		{
			activation[Di]=Element[Di].getActivation();
		}
		return activation;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public double getActivation(int Vi) 
	{
		if(Element.length<=Vi)    
		{
			return 0;
		}
		return Element[Vi].getActivation();
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void setActivation(int Vi, double value) 
	{
		Element[Vi].setActivation(value);
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public long getActivitySignal()
	{
		return 1;
	}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public synchronized double[][] getRelationalDerivative() {double[][] derivative = new double[Element.length][]; for(int i=0; i<Element.length; i++) {derivative[i] = Element[i].getRelationalDerivative();}return derivative;}

	public synchronized double getRelationalDerivative(int Vi, int Ii) {if(Vi<0) {return 0;}return Element[Vi].getRelationalDerivative()[Ii];}

	public void setRelationalDerivative(double[][] derivative) {for(int Vi=0; Vi<this.size()&&Vi<derivative.length; Vi++) {Element[Vi].setRelationalDerivative(derivative[Vi]);}}

	public boolean hasRelationalDerivative(int Vi) {if(Element[Vi].getRelationalDerivative()!=null) {return true;}return false;}

	public double[] getRelationalDerivative(int Vi) {if(Vi<0) {return null;}return Element[Vi].getRelationalDerivative();}

	public void setRelationalDerivative(int Vi, double[] gradient) {Element[Vi].setRelationalDerivative(gradient);}

	public synchronized void setRelationalDerivative(int Vi, int Ii, double value) {Element[Vi].setRelationalDerivative(Ii, value);}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public double[][] getInputDerivative() {double[][] derivative = new double[Element.length][]; for(int i=0; i<Element.length; i++) {derivative[i] = Element[i].getInputDerivative();}return derivative;}

	public double getInputDerivative(int Vi, int Ii) {if(Vi<0) {return 0;}return Element[Vi].getInputDerivative()[Ii];}

	public void setInputDerivative(double[][] derivative) {for(int Vi=0; Vi<this.size()&&Vi<derivative.length; Vi++) {Element[Vi].setInputDerivative(derivative[Vi]);}}

	public boolean hasInputDerivative(int Vi) {if(Element[Vi].getInputDerivative()!=null) {return true;}return false;}

	public double[] getInputDerivative(int Vi) {if(Vi<0) {return null;}return Element[Vi].getInputDerivative();}

	public void setInputDerivative(int Vi, double[] gradient) {Element[Vi].setInputDerivative(gradient);}

	public synchronized void setInputDerivative(int Vi, int Ii, double value) {Element[Vi].setInputDerivative(Ii, value);}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public synchronized double[][] getInput() {double[][] input = new double[Element.length][]; for(int i=0; i<Element.length; i++) {input[i] = Element[i].getInput();}return input;}

	public synchronized double getInput(int Vi, int Ii) {if(Vi<0) {return 0;}return Element[Vi].getInput()[Ii];}

	public synchronized void setInput(double[][] input) {for(int i=0; i<Element.length&&i<input.length; i++) {Element[i].setInput(input[i]);}}

	public boolean hasInput(int Vi) {if(Element[Vi].getInput()!=null) {return true;}return false;}

	public synchronized void setInput(int Vi, double[] input) {Element[Vi].setInput(input);}

	public synchronized void setInput(int Vi, int Ii, double value){Element[Vi].setInput(Ii, value);}

	public synchronized void setAllInput(int index, double input) {for(int i=0; i<Element.length; i++) {Element[i].setInput(index, input);}}

	public synchronized void addToAllInput(int index, double input) {for(int i=0; i<Element.length; i++) {Element[i].addToInput(index, input);}}

	public synchronized void addToInput(double[][] input) {for(int i=0; i<Element.length&&i<input.length; i++) {for(int Ii=0; Ii<Element[i].Variable.length; Ii++) {Element[i].addToInput(Ii, input[i][Ii]);}}}

	public double[] getInput(int Vi) {if(Vi<0) {return null;}return Element[Vi].getInput();}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//@Override// NVNode
	public boolean 	  hasBiasGradient(int Vi, int Ii) {if(Vi==0) {if(Element[Vi].Variable!=null) {if(Element[Vi].Variable.length>Ii&&Element[Vi].Variable[Ii].hasBiasGradient()) {return true;}}}return false;}
	//@Override// NVNode
	public double     getBiasGradient(int Vi, int Ii) {if(Vi==0) {if(Element[Vi].Variable!=null) {if(Element[Vi].Variable.length>Ii) {return Element[Vi].Variable[Ii].BiasGradient;}}}return 0;}
	//@Override// NVNode
	public void	      setBiasGradient(int Vi, int Ii, double value) {if(Vi==0) {if(Element[Vi].Variable!=null) {if(Element[Vi].Variable.length>Ii) {Element[Vi].Variable[Ii].BiasGradient=value;}}}}
	//@Override// NVNode
	public void	      addToBiasGradient(int Vi, int Ii, double value) {if(Vi==0) {if(Element[Vi].Variable!=null) {if(Element[Vi].Variable.length>Ii) {Element[Vi].Variable[Ii].BiasGradient+=value;}}}}
	//-------------------------------------------------------------
	//@Override// NVNode
	public boolean 	  hasBiasGradient(int Ii) {if(Element[0].Variable[Ii].hasBiasGradient()) {return true;}return false;}
	//@Override// NVNode
	public double[]   getBiasGradient(int Vi) {if(Vi>=0) {if(Element.length>Vi) {return Element[Vi].BiasGradient();}}return null;}
	//@Override// NVNode
	public void	      setBiasGradient(int Vi, double[] array) {if(Vi==0) {for(int i=0; i<Element[Vi].Variable.length&&i<array.length; i++) {Element[Vi].Variable[i].BiasGradient=array[i];}}}
	//@Override// NVNode
	public void	      addToBiasGradient(int Vi, double[] array) {if(Vi==0) {for(int i=0; i<Element[Vi].Variable.length&&i<array.length; i++) {Element[Vi].Variable[i].BiasGradient+=array[i];}}}
	//-------------------------------------------------------------
	//@Override// NVNode
	public boolean 	  hasBiasGradient() {for(int Ii=0; Ii<this.inputSize(); Ii++) {if(Element[0].hasBiasGradient(Ii)) {return true;}}return false;}
	//@Override// NVNode
	public double[][] getBiasGradient() {double[][] newBiasGradient = new double[Element.length][];for(int Di=0; Di<Element.length; Di++) {newBiasGradient[Di]=Element[Di].BiasGradient();} return newBiasGradient;}
	//@Override// NVNode
	public void	      setBiasGradient(double[][] array) {if(array==null) {for(int Vi=0; Vi<Element.length; Vi++) {for(int Ii=0; Ii<Element[Vi].Variable.length; Ii++) {Element[Vi].Variable[Ii].BiasGradient=null;}}}else{for(int Vi=0; Vi<Element.length&&Vi<array.length; Vi++) {this.setBiasGradient(Vi, array[Vi]);}}}
	//@Override// NVNode
	public void	      addToBiasGradient(double[][] array) {if(array!=null) {for(int Vi=0; Vi<Element.length&&Vi<array.length; Vi++) {this.addToBiasGradient(Vi, array[Vi]);}}}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//@Override // NVNode
	public boolean 	  hasBias(int Vi, int Ii) {if(Vi==0) {if(Element[Vi].Variable!=null) {if(Element[Vi].Variable.length>Ii) {return true;}}}return false;}
	//@Override // NVNode
	public double     getBias(int Vi, int Ii) {if(Vi==0) {if(Element[Vi].Variable!=null) {if(Element[Vi].Variable.length>Ii) {return Element[Vi].Variable[Ii].Bias;}}}return 0;}
	//@Override // NVNode
	public void	      setBias(int Vi, int Ii, double value) {if(Vi==0) {if(Element[Vi].Variable!=null) {if(Element[Vi].Variable.length>Ii) {Element[Vi].Variable[Ii].Bias=value;}}}}
	//@Override // NVNode
	public void	      addToBias(int Vi, int Ii, double value) {if(Vi==0) {if(Element[Vi].Variable!=null) {if(Element[Vi].Variable.length>Ii) {Element[Vi].Variable[Ii].Bias+=value;}}}}
	//-------------------------------------------------------------
	//@Override
	public boolean 	  hasBias(int Ii) {if(Element[0].Variable[Ii].hasBias()) {return true;}return false;}
	//@Override // NVNode
	public double[]   getBias(int Vi) {if(Vi>=0) {if(Element.length>Vi) {return Element[Vi].Bias();}}return null;}
	//@Override // NVNode
	public void	      setBias(int Vi, double[] array) {if(Vi==0) {for(int i=0; i<Element[Vi].Variable.length&&i<array.length; i++) {Element[Vi].Variable[i].Bias=array[i];}}}
	//@Override // NVNode
	public void	      addToBias(int Vi, double[] array) {if(Vi==0) {for(int i=0; i<Element[Vi].Variable.length&&i<array.length; i++) {Element[Vi].Variable[i].Bias+=array[i];}}}
	//-------------------------------------------------------------
	//@Override // NVNode
	public boolean 	  hasBias() {for(int Ii=0; Ii<this.inputSize(); Ii++) {if(Element[0].hasBias(Ii)) {return true;}}return false;}
	//@Override // NVNode
	public double[][] getBias() {double[][] newBias = new double[Element.length][];for(int Di=0; Di<Element.length; Di++) {newBias[Di]=Element[Di].Bias();} return newBias;}
	//@Override // NVNode
	public void	      setBias(double[][] array) {if(array==null) {for(int Vi=0; Vi<Element.length; Vi++) {for(int Ii=0; Ii<Element[Vi].Variable.length; Ii++) {Element[Vi].Variable[Ii].Bias=null;}}}else{for(int Vi=0; Vi<Element.length&&Vi<array.length; Vi++) {this.setBias(Vi, array[Vi]);}}}
	//@Override // NVNode
	public void	      addToBias(double[][] array) {if(array!=null) {for(int Vi=0; Vi<Element.length&&Vi<array.length; Vi++) {this.addToBias(Vi, array[Vi]);}}}
	//-------------------------------------------------------------
	//@Override// NVNode
	public boolean hasWeightGradient(int Vi, int Ii, int Ui) {if(hasWeightGradient(Vi,Ii)) {if(Element[Vi].Variable[Ii].WeightGradient.length>Ui&&Ui>=0) {return true;}}return false;}
	//@Override// NVNode
	public double  getWeightGradient(int Vi, int Ii, int Ui) {if (hasWeightGradient(Vi, Ii, Ui)) {return Element[Vi].Variable[Ii].WeightGradient[Ui];}return 0;}
	//@Override// NVNode
	public void    setWeightGradient(int Vi, int Ii, int Ui, double value) {if (hasWeightGradient(Vi, Ii, Ui)) {Element[Vi].Variable[Ii].WeightGradient[Ui]=value;}}
	//@Override// NVNode
	public void    addToWeightGradient(int Vi, int Ii, int Ui, double value) {if (hasWeightGradient(Vi, Ii, Ui)) {Element[Vi].Variable[Ii].WeightGradient[Ui]+=value;}}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//@Override // NVNode
	public boolean  hasWeightGradient(int Vi, int Ii) {if(hasWeightGradient(Vi)) {if(Element[Vi].Variable.length>Ii&&Ii>=0) {if(Element[Vi].Variable[Ii].WeightGradient!=null) {return true;}}}return false;}
	//@Override// NVNode
	public double[] getWeightGradient(int Vi, int Ii) {if(hasWeightGradient(Vi,Ii)) {return Element[Vi].Variable[Ii].WeightGradient;}return null;}
	//@Override// NVNode
	public void     setWeightGradient(int Vi, int Ii, double[] array) {if(Element[Vi].Variable[Ii]!=null) {Element[Vi].Variable[Ii].WeightGradient=array;}}
	//@Override// NVNode
	public void     addToWeightGradient(int Vi, int Ii, double[] array) {if(Element[Vi].Variable[Ii]!=null) {for(int i=0; i<Element[Vi].Variable[Ii].WeightGradient.length&&i<array.length; i++) {Element[Vi].Variable[Ii].WeightGradient[i]=array[i];}}}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//@Override// NVNode
	public boolean    hasWeightGradient(int Vi) {if(Vi<Element.length) {if(Element[Vi].Variable[0].WeightGradient!=null) {return true;}}return false;}
	//@Override// NVNode
	public double[][] getWeightGradient(int Vi) {return Element[Vi].WeightGradient();}
	//@Override// NVNode
	public void       setWeightGradient(int Vi, double[][] array) {if(array==null) {array = new double[this.inputSize()][];}for(int Ii=0; Ii<Element[Vi].Variable.length&&Ii<array.length; Ii++) {setWeightGradient(Vi,Ii,array[Ii]);}}
	//@Override// NVNode
	public void       addToWeightGradient(int Vi, double[][] array) {if(array==null) {array = new double[this.inputSize()][];}for(int i=0; i<Element[Vi].Variable.length&&i<array.length; i++) {addToWeightGradient(Vi,i,array[i]);}}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//@Override// NVNode
	public boolean 		hasWeightGradient() {if(Element[0].Variable[0].WeightGradient!=null) {return true;}return false;}
	//@Override// NVNode
	public double[][][] getWeightGradient() {double[][][] array = new double[Element.length][][];for(int i=0; i<array.length; i++) {array[i]=Element[i].WeightGradient();}return array;}
	//@Override// NVNode
	public void		    setWeightGradient(double[][][] array) {if(array==null) {array = new double[this.size()][][];}for(int Vi=0; Vi<array.length&&Vi<Element.length; Vi++) {this.setWeightGradient(Vi, array[Vi]);}}
	//@Override// NVNode
	public void		    addToWeightGradient(double[][][] array) {if(array==null) {array = new double[this.size()][][];}for(int Vi=0; Vi<array.length&&Vi<Element.length; Vi++) {this.addToWeightGradient(Vi, array[Vi]);}}
	//-------------------------------------------------------------
	//@Override// NVNode
	public boolean hasWeight(int Vi, int Ii, int Wi) {if(hasWeight(Vi,Ii)) {if(Element[Vi].Variable[Ii].Weight.length>Wi&&Wi>=0) {return true;}}return false;}
	//@Override// NVNode
	public double  getWeight(int Vi, int Ii, int Wi) {if (hasWeight(Vi, Ii, Wi)) {return Element[Vi].Variable[Ii].Weight[Wi];}return 0;}
	//@Override// NVNode
	public void    setWeight(int Vi, int Ii, int Wi, double value) {if (hasWeight(Vi, Ii, Wi)) {Element[Vi].Variable[Ii].Weight[Wi]=value;}}
	//@Override// NVNode
	public void    addToWeight(int Vi, int Ii, int Wi, double value) {if (hasWeight(Vi, Ii, Wi)) {Element[Vi].Variable[Ii].Weight[Wi]+=value;}}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//@Override // NVNode
	public boolean  hasWeight(int Vi, int Ii) {if(Vi<Element.length&&Vi>=0) {if(Element[Vi].Variable.length>Ii&&Ii>=0) {if(Element[Vi].Variable[Ii].Weight!=null) {return true;}}}return false;}
	//@Override// NVNode
	public double[] getWeight(int Vi, int Ii) {if(hasWeight(Vi,Ii)) {return Element[Vi].Variable[Ii].Weight;}return null;}
	//@Override// NVNode
	public void     setWeight(int Vi, int Ii, double[] array) {if(Element[Vi].Variable[Ii]!=null) {Element[Vi].Variable[Ii].Weight=array;}}
	//@Override// NVNode
	public void     addToWeight(int Vi, int Ii, double[] array) {if(Element[Vi].Variable[Ii]!=null) {for(int i=0; i<Element[Vi].Variable[Ii].Weight.length&&i<array.length; i++) {Element[Vi].Variable[Ii].Weight[i]=array[i];}}}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//@Override// NVNode
	public boolean    hasWeight(int Vi) {if(Vi<Element.length) {if(Element[Vi].Variable[0].Weight!=null) {return true;}}return false;}
	//@Override// NVNode
	public double[][] getWeight(int Vi) {return Element[Vi].Weight();}
	//@Override// NVNode
	public void       setWeight(int Vi, double[][] array) {if(array==null) {array = new double[this.inputSize()][];}for(int Ii=0; Ii<Element[Vi].Variable.length&&Ii<array.length; Ii++) {setWeight(Vi,Ii,array[Ii]);}}
	//@Override// NVNode
	public void       addToWeight(int Vi, double[][] array) {if(array==null) {array = new double[this.inputSize()][];}for(int i=0; i<Element[Vi].Variable.length&&i<array.length; i++) {addToWeight(Vi,i,array[i]);}}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//@Override// NVNode
	public boolean 		hasWeight() {if(Element[0].Variable[0].Weight!=null) {return true;}return false;}
	//@Override// NVNode
	public double[][][] getWeight() {double[][][] array = new double[Element.length][][];for(int i=0; i<array.length; i++) {array[i]=Element[i].Weight();}return array;}
	//@Override// NVNode
	public void		    setWeight(double[][][] array) {if(array==null) {array = new double[this.size()][][];}for(int Vi=0; Vi<array.length&&Vi<Element.length; Vi++) {this.setWeight(Vi, array[Vi]);}}
	//@Override// NVNode
	public void		    addToWeight(double[][][] array) {if(array==null) {array = new double[this.size()][][];}for(int Vi=0; Vi<array.length&&Vi<Element.length; Vi++) {this.addToWeight(Vi, array[Vi]);}}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public void weightAll()
	{
		for(int Vi=0; Vi<this.size(); Vi++) {
			for(int Ii=0; Ii<this.inputSize(); Ii++) {
				if(Element[Vi].Variable[Ii].hasWeight()==false) {
					int contextSize = 0;
					for(int Ni=0; Ni<Element[Vi].Variable[Ii].size(); Ni++) {contextSize++;}
					Element[Vi].Variable[Ii].Weight = new double[contextSize];
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
				Element[Vi].Variable[Ii].Weight = null;
			}
		}
	}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public synchronized void freeInput()
	{
		for(int i=0; i<Element.length; i++)
		{
			Element[i].setInput(null);
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
		return Element[0].Variable.length;
	}
	//--------------------------------------
	public String toString(){
		String result = "";
		String level = "    ";
		for(int Vi=0; Vi<this.size(); Vi++){
			result += toString(Vi);
		}
		return result;
	}

	public String toString(int Vi){
		String base = "||    ";
		String result = "";
		String level = "    ";
		result += "[]=|D|=|A|=|T|=|A|==|F|=|I|=|E|=|L|=|D|=|S|==>\n"+base+"\n";
		result += base+"[Element]["+Vi+"]:\n"+base+"(\n";
		result += base+level+"[Activation]:("+this.getActivation(Vi)+"),\n";
		result += base+level+"[Error]:("+this.getError(Vi)+"),\n";
		result += base+level+"[Optimum]:("+this.getOptimum(Vi)+"),\n";
		for(int Ii=0; Ii<Element[Vi].Variable.length; Ii++){
			result += base+level+"[Input]["+Ii+"]:(\n";
			result += base+level+level+"[Value]:("+Element[Vi].Variable[Ii].Input+"),\n";
			result += base+level+level+"[Derivative]:("+Element[Vi].Variable[Ii].InputDerivative+"),\n";
			result += base+level+level+"[Bias]:("+Element[Vi].Variable[Ii].Bias+"),\n";
			result += base+level+level+"[BiasGradient]:("+Element[Vi].Variable[Ii].BiasGradient+"),\n";
			result += base+level+level+"[Weight]:(\n"+base+level+level+level;
			if(Element[Vi].Variable[Ii].Weight!=null){
				for(int Wi=0; Wi<Element[Vi].Variable[Ii].Weight.length; Wi++){
					result += "["+Wi+"]:("+Element[Vi].Variable[Ii].Weight[Wi]+"), ";
				}
			}else{
				result+="null";
			}
			result += "\n"+base+level+level+"),\n";
			result += base+level+level+"WeightGradient:(\n"+base+level+level+level;
			if(Element[Vi].Variable[Ii].WeightGradient!=null){
				for(int Wi=0; Wi<Element[Vi].Variable[Ii].Weight.length; Wi++){
					result += "["+Wi+"]:("+Element[Vi].Variable[Ii].WeightGradient[Wi]+"), ";
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
