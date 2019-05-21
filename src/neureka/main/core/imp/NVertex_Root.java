package neureka.main.core.imp;

import java.math.BigInteger;

import neureka.main.core.NVChild;
import neureka.main.core.NVNeuron;
import neureka.main.core.NVNode;
import neureka.main.core.NVParent;
import neureka.main.core.NVRoot;
import neureka.main.core.NVUtility;
import neureka.main.core.NVertex;
import neureka.main.core.modul.calc.NVFunction;
import neureka.main.core.base.NVData;
import neureka.main.core.modul.mem.NVMemory;
import neureka.main.core.modul.opti.NVOptimizer;
import neureka.main.exec.NVExecutable;
import neureka.ngui.NPanelNode;

public class NVertex_Root extends NVertex implements NVNeuron, NVRoot, NVExecutable, NVChild, NVParent
{
	private NVRoot Parent;
	private long Signal;
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public NVertex_Root(int size, boolean isFirst, boolean isLast, int neuronID, int functionID)
	{
		super(size, isFirst, isLast, neuronID, functionID);
		shoutLine("new NVertex_Root(int size="+size+", boolean isFirst="+isFirst+", boolean isLast="+isLast+", int v_id="+neuronID+", int f_id="+functionID+")|: ...created!");
		setParent((NVParent)this);
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public NVertex_Root(int size, boolean isFirst, boolean isLast, int neuronID, int functionID, int tunnel)
	{
		super(size, isFirst, isLast, neuronID, functionID, tunnel);
		shoutLine("new NVertex_Root(int size="+size+", boolean isFirst="+isFirst+", boolean isLast="+isLast+", int v_id="+neuronID+", int f_id="+functionID+", int delay="+tunnel+")|: ...created!");
		setParent((NVParent)this);
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public NVertex_Root(NVNode parent, boolean isFirst, boolean isLast, int neuronID, int functionID)
	{
		super(parent, isFirst, isLast, neuronID, functionID, 1);
		shoutLine("new NVertex_Root(NVNode mother=#"+parent.asCore().getID()+", boolean isFirst="+isFirst+", boolean isLast="+isLast+", int v_id="+neuronID+", int f_id="+functionID+")|: ...created!");
		if(parent==null){
			setParent((NVParent)this);
		}
		else{
			setParent(parent.asRoot().asParent());
	    }
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public NVertex_Root(NVNode parent, boolean isFirst, boolean isLast, int neuronID, int functionID, int tunnel)
	{
		super(parent, isFirst, isLast, neuronID, functionID, tunnel);
		shoutLine("new NVertex_Root(NVNode mother=#"+parent.asCore().getID()+", boolean isFirst="+isFirst+", boolean isLast="+isLast+", int v_id="+neuronID+", int f_id="+functionID+", int delay="+tunnel+")|: ...created!");
		if(parent==null){
			setParent((NVParent)this);
		}
		else{
			setParent(parent.asRoot().asParent());
	    }
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public NVertex_Root(NVertex toBeCopied)
	{
		super(toBeCopied);
		setParent(this);
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NExecutable
	public void forward()
	{
		shoutLine("NVertex_Root->forward()|: ...");
		shoutLine("========================================================================");
		//if this is e_sub root return else startRootForward!
		if(this.is(NVertex.Child))
		{
			shoutLine("-forward()|: (this == child): => return");
			shoutLine("-forward()|: END\n");
			return;
		}
		NVData result = null;
		boolean[] signal = saved(publicized(inputSignalIf(forwardCondition())));

		shoutLine("-forward()|: for(int Vi=0; Vi<this.size(); Vi++):");
		for(int Vi=0; Vi<this.size(); Vi++)
		{
			shoutLine("-forward()|: 'Vi':"+Vi+"");
			shoutLine("-forward()|: ->activationOf(weightConvectionOf(publicized(inputSignalIf(forwardCondition())),"+Vi+")"+Vi+");");
			result = //THIS IS WORK IN PROGRESS!!!
			activationOf
			(
				weightedConvectionOf(result, signal, Vi),
				Vi
		    );
		}
		memorize(result);
		shoutLine("-forward()|: END\n");
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override //NExecutable
	public synchronized boolean forwardCondition()
	{
		shoutLine("");
		shoutLine("NVertex_Root->forwardCondition()|: ...");
		shoutLine("========================================================================");
		if(!is(NVertex.TrainableNeuron) && !is(NVertex.MotherRoot) && !this.is(NVertex.RootMemory))
		{
			shoutLine("-forwardCondition()|: This is dependend e_sub-root node. Node cannot perform forward propagation on its own.");
			shoutLine("-inputSignalIf(...)|: => return false;");
			shoutLine("-inputSignalIf(...)|: => END\n ");
			return false;
		}
		//if this is memory root node -> adjust your synchronization object
		if (isParalyzed() == true)
		{
			shoutLine("-forwardCondition()|: This Neuron is not functional! No weight convection!");
			shoutLine("-inputSignalIf(...)|: => return false;");
			shoutLine("-inputSignalIf(...)|: => END\n ");
			return false;
		}
		shoutLine("-inputSignalIf(...)|: return true;");
 	    shoutLine("-inputSignalIf(...)|: END\n ");
		return true;
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NExecutable
	public boolean[] inputSignalIf(boolean condition)
	{
		shoutLine("NVertex_Root->inputSignalIf(boolean condition)|: ...");
		shoutLine("========================================================================");
		boolean[] signal;
		if (Connection == null)
		{
			shoutLine("-inputSignalIf(...)|: (Connection == null):");
			shoutLine("-inputSignalIf(...)|: => signal = new boolean[1]; ");
			shoutLine("-inputSignalIf(...)|: => signal[0]=false; ");
			shoutLine("-inputSignalIf(...)|: => return signal; ");
			shoutLine("-inputSignalIf(...)|: => END\n ");
		 	signal = new boolean[1];
		 	signal[0]=false;
		 	return signal;
		}
		signal = new boolean[Connection.length];
		if(isFirst())
		{
			shoutLine("-inputSignalIf(...)|: (isFirst() == true):");
			shoutLine("-inputSignalIf(...)|: => for(int Ii=0; Ii<signal.length; Ii++) {signal[Ii]=true;}");
			for(int Ii=0; Ii<signal.length; Ii++)
			{
				shoutLine("-inputSignalIf(...)|: =>('Ii':"+Ii+")|: signal[Ii]=true;");
				signal[Ii]=true;
			}
			shoutLine("-inputSignalIf(...)|: => return signal;");
			shoutLine("-inputSignalIf(...)|: => END\n");
			return signal;
		}//firstlayer neurons are always signaling!
		shoutLine("-inputSignalIf(...)|: long selfState = 'ActivitySignal()':"+getActivitySignal()+";");
		long selfState = getActivitySignal();
		shoutLine("-inputSignalIf(...)|: for(int Ii = 0; Ii < Connection.length; Ii++):");
		for (int Ii = 0; Ii < Connection.length; Ii++)
		{
			shoutLine("-inputSignalIf(...)|: ('Ii':"+Ii+")|: ...");
			if (Connection[Ii] != null)
			{
				if(isRootConnection(Connection[Ii])==false)
				{
					shoutLine("-inputSignalIf(...)|: ('Ii':"+Ii+")|:(Connection[Ii] != null && isRootConnection(Connection[Ii])==false):");
 				shoutLine("-inputSignalIf(...)|: ('Ii':"+Ii+")|:=> signal[Ii]=false;");
 				signal[Ii]=false;
 				shoutLine("-inputSignalIf(...)|: ('Ii':"+Ii+")|:=> for (int Ni = 0; Ni < Connection[Ii].length; Ni++):");
 				for (int Ni = 0; Ni < Connection[Ii].length; Ni++)
 				{
 					shoutLine("-inputSignalIf(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|: long otherState = Connection[Ii][Ni].ActivitySignal();");
 					long otherState = Connection[Ii][Ni].getActivitySignal();
 					shoutLine("-inputSignalIf(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|: long otherState = Connection[Ii][Ni].ActivitySignal(); ");
 				    shoutLine("-inputSignalIf(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|: 'otherState':"+otherState+", 'selfState': "+selfState);
 					//--------------------------------------------------
 					if(otherState>0 || selfState<otherState)
 					{
 						shoutLine("-inputSignalIf(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:(otherState > 0 || selfState < otherState):");
 						shoutLine("-inputSignalIf(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:=> signal[counter]=true; Ni=Connection[Ii].length;");
 						signal[Ii]=true;
 						Ni=Connection[Ii].length;
 					}
 					/* TIMELINES:
 					 * =========
 					 * Other: |ooo----o---|(-3) -> 3 steps inactive!
 					 * Self:  |o-ooooo----|(-4) -> 4 steps inactive!
 					 * =====================================
 					 * ->The other node (Connection) has been active -> however activation was inhibited (paralyzed)
 					 * ->Therefore activation will now take place! (with a delay)
 					 * */
 				}
				}
				else
				{
					boolean check = false;
					for (int Ni = 0; Ni < Connection[Ii].length; Ni++)
 				{
					//Connection[Ii][Ni].asRoot().asChild().
						boolean childSignal = // Correct to distributeInputSignal
								Connection[Ii][Ni].asRoot().distributeRootSignal(selfState);
						if(childSignal==true) {
							check = true;
						}
 				}
					signal[Ii]=check;
				}
			}
			else
			{
				shoutLine("-inputSignalIf(...)|: ('Ii':"+Ii+")|:((Connection[Ii] != null && isRootConnection(Connection[Ii])==false)==false):");
				shoutLine("-inputSignalIf(...)|: ('Ii':"+Ii+")|:=> I[i"+Ii+"] has no connections to check!");
				shoutLine("-inputSignalIf(...)|: ('Ii':"+Ii+")|:=> signal[Ii] = false;");
				signal[Ii]=false;
			}
		}
		shoutLine("-inputSignalIf(...)|: for(int Si=0; Si<signal.length; Si++):");
		for(int Si=0; Si<signal.length; Si++)
		{
			shoutLine("-inputSignalIf(...)|: ('Si':"+Si+")|: 'signal["+Si+"]':"+signal[Si]+"; ");
		}
		shoutLine("-inputSignalIf(...)|: END\n");
		return signal;
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	   @Override //NRoot
	public boolean distributeRootSignal(long phaseSignal)
	{
		shoutLine("NVertex_Root->distributeRootSignal(long phaseSignal)|: ...");
		shoutLine("========================================================================");
		if(State.isRoot(this) == false)
		{
			shoutLine("-distributeRootSignal(...)|: (State.isRoot(this) == false):");
			shoutLine("-distributeRootSignal(...)|: => return true;");
			shoutLine("-distributeRootSignal(...)|: => END\n");
			return true;// Outside of root!
		}
		if(Connection==null)
		{
			shoutLine("-distributeRootSignal(...)|: (Connection==null):");
			shoutLine("-distributeRootSignal(...)|: => return false;");
			shoutLine("-distributeRootSignal(...)|: => END\n");
			return false;
		}
		shoutLine("-distributeRootSignal(...)|: boolean activity = false;");
		boolean activity = false;
		shoutLine("-distributeRootSignal(...)|: for(int Ii=0; Ii<Connection.length; Ii++): ...");
		for(int Ii=0; Ii<Connection.length; Ii++)
		{
			shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|: ...");
			shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|: boolean currentActivity=false;");
			boolean currentActivity=false;
			if(Connection[Ii]!=null)
			{
				shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:(Connection[Ii]!=null):");
				shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=> for(int Ni=0; Ni<Connection[Ii].length; Ni++):");
				for(int Ni=0; Ni<Connection[Ii].length; Ni++)
				{
					shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|: Connection[Ii][Ni]=Connection[Ii][Ni].Node();");
					Connection[Ii][Ni]=Connection[Ii][Ni].asNode();
					if(Connection[Ii][Ni].is(Child))
					{
						shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:(Connection[Ii][Ni].is(Child)):");
						shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:=> boolean newSignal = Connection[Ii][Ni].Core().Root().distributeRootSignal(phaseSignal);");
						boolean newSignal = Connection[Ii][Ni].asCore().asRoot().distributeRootSignal(phaseSignal);
						if(newSignal)
						{
							shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:=>(newSignal):");
							shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:=>=> activity = true;");
							shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:=>=> currentActivity=true;");
							activity = true;
							currentActivity=true;
							shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:=>|>");
						}
						shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:|>");
					}
					else
					{
						shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:(Connection[Ii][Ni].is(Child)==false):");
						shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:=> long otherState = Connection[Ii][Ni].ActivitySignal();");
					    long otherState = Connection[Ii][Ni].getActivitySignal();
					    shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:=>  'otherState':"+otherState+", 'phaseSignal': "+phaseSignal);
						//--------------------------------------------------
						if(otherState>0 || phaseSignal<otherState)
						{
							shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:=>(otherState>0 || phaseSignal<otherState):");
							shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:=>=> activity=true; ");
							shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:=>=> currentActivity=true;");
							shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:=>=> Ni=Connection[Ii].length;");
						    activity=true;
						    currentActivity=true;
						    Ni=Connection[Ii].length;
						    shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:=>|>");
						}
						/*TIMELINES:
						 *
						 *Other: |ooo----o---|(-3) -> 3 steps inactive!
					     *Self:  |o-ooooo----|(-4) -> 4 steps inactive!
				         *
					     *->The other node (Connection) has been active -> however activation was inhibited (paralyzed)
						 *->Therefore activation will now take place! (with a delay)
						 * */
						shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:|>");
					}
				}
				shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|:|>");
			}
			shoutLine("-distributeRootSignal(...)|: ('Ii':"+Ii+")|: updateInputState(currentActivity, Ii);");
			updateInputState(currentActivity, Ii);
			//mother.registerInputActivity()...-> if mother has NVConvector -> activity filter is being created...
		}
		shoutLine("-distributeRootSignal(...)|: return 'activity':"+activity+";");
		shoutLine("-distributeRootSignal(...)|: END\n");
		return activity;
	}//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//@Override //NExecutable
	 public boolean[] saved(boolean[] signal)
	 {
	 	shoutLine("NVertex_Root->saved(boolean[] signal)|: ...");
		shoutLine("========================================================================");
		if(signal==null)
		{
			shoutLine("-saved(...)|: (signal==null):=> return null; END\n");
			return null;
		}
		shoutLine("-saved(boolean[] signal)|: for(int Ii=0; Ii<signal.length&&Ii<this.InputState.length; Ii++)");
		for(int Ii=0; Ii<signal.length && Ii<this.InputState.length; Ii++)
		{
			shoutLine("-saved(boolean[] signal)|: ('Ii':"+Ii+")|: updateInputState(signal[Ii], Ii);");
			updateInputState(signal[Ii], Ii);
		}
		shoutLine("-saved(...)|: END\n");
		return signal;
	}
	 //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public boolean[] publicized(boolean[] signal) // @CONSTANT
	{
		shoutLine("NVertex_Root->publicized(boolean[] signal)|: ...");
		shoutLine("========================================================================");
		if(signal==null)
		{
			shoutLine("-publicized(...)|: (signal==null):=> return null; END\n");
			return null;
		}
		NPanelNode PanelNode = (NPanelNode) findModule(NPanelNode.class);
		if (PanelNode != null)
		{
			if (PanelNode != null){
				PanelNode.requestDataDisplayUpdate();
				PanelNode.setInputActivity(signal);//This signal is not complete!
			} // IO DISPLAY!!!!
		}
		shoutLine("-publicized(...)|: END\n");
		return signal;
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NExecutable
	public boolean isRootConnection(NVNode[] Connection)
	{
		if(Connection!=null)
		{
			if(Connection[0].is(NVertex.Child)) {
				return true;
			}
		}
		return false;
	}
	@Override
	public NVData weightedConvectionOf(NVData Data, boolean[] Signal, int Vi)
	{
		shoutLine("NVertex_Root->startWeightConvection(boolean[] inputSignal, int Vi)|: ...");
		shoutLine("========================================================================");
		//Dim = findModule(...);
		int[] selfDim, otherDim, weightDim, form;
		//int Vi, int Wi
		/**
		 * 		int[] selfDim = findModule(int[].class);
		 *
		 * 		//[Ii][Ni][0]=>weightDim [1]=>form
		 *		int[][] FormData = findModule(int[][][][].class)[Ii][Ni]
		 *	    int[] weightDim = FormData[0];
		 *	    int[] connForm = FormData[1];
		 *
		 * */
		if(Data==null) {
			Data = new NVData(Neuron);
		}
		if(Data.hasInput(Vi)==false) {//maybe return instead?
			Data.setInput(Vi, new double[Connection.length]);
		}
		if(Signal==null)
		{
			Signal=new boolean[this.inputSize()];
			for(int Si=0; Si<Signal.length; Si++)
			{
				Signal[Si]=true;
			}
		}
		else
		{
			if(Signal.length!=inputSize())
			{
				boolean[] newSignal = new boolean[this.inputSize()];
				for(int Si=0; Si<Signal.length && Si<newSignal.length; Si++)
				{
					newSignal[Si]=Signal[0];
				}
				Signal = newSignal;
			}
		}
		double[] update = {0};
		boolean[] isRootConnection = {false};
		//===============================================================================
		QuinIteration ConvectionStep =
 		(int vi, int ii, int ni, int ei, int wi)->
		{
	    		if(Connection[ii][ni]!=asNode() && isRootConnection[0])
	    		{
	    			if (this.hasWeight(vi, ii)) {// WHY? => inner root nodes dont need weighted connections
	    			  	update[0] += Connection[ii][ni].asRoot().findRootActivation(vi) * this.getWeight(vi, ii, wi);
	    			}
	    			else {//If connection == this ==> bn
	    			  	update[0] += Connection[ii][ni].asRoot().findRootActivation(vi) * 1;
	    			}
	    		}
	    		else
	   		{
	   			if(this.hasWeight(vi, ii)) {//this.hasWeight(vi, ii) WHY? :=> inner root nodes don't need weighted connections
	    			  	update[0] += (Connection[ii][ni].getActivation(ei)-Connection[ii][ni].getOptimum(ei)) * this.getWeight(vi, ii, wi);
	    			}
	    			else {//If connection == this ==> no need for recursive call
	    			  	update[0] += (Connection[ii][ni].getActivation(ei)-Connection[ii][ni].getOptimum(ei)) * 1;
	    			}
	    		}
		};
 		//===============================================================================
		for (int Ii = 0; Ii < Connection.length; Ii++)
		{
			if(Signal[Ii]==true && Connection[Ii]!=null) {//Note: -> if an input does not recieve a signal-> the last input will be retained! (resetting it would distort mathematical consistancy)
				Data.setInput(Vi, Ii, 0.0);
				isRootConnection[0] = (State.isChild(Connection[Ii][0].asNode())) ? true : false;
			    if(isRootConnection[0])
			    {
					for (int Ni = 0; Ni < Connection[Ii].length; Ni++)
				    {
						update[0] = 0;//Note: Vi == Wi (Wi not needed!)
				        //==================================================
				     	ConvectionStep.on(Vi, Ii, Ni, Ni, Ni);
				     	//==================================================
				     	Data.addToInput(Vi, Ii, update[0]);
				    }
				}
				else
				{
					int Wi=0;
					for (int Ni = 0; Ni < Connection[Ii].length; Ni++)
				    {
				    	//otherDim = Connection[Ii][Ni]
						//wightDim = findModule()
						//OtoWFrmt = Connection[Ii][Ni].frmt
						update[0] = 0;
				     	int limit = Connection[Ii][Ni].size()+Wi;
				     	int Ci=0;
				     	while(Wi<limit)//<= Cycling through weights of current node!
			        	{
			        		//TODO: self.Dim[ 3][2  ][ 5][4  ][2]=>Vi to coord!
							//TODO: seWo.Dim[-1][  2][-4][  2]   =>Wi to coord!
							//TODO: rslt.Dim[ 3][2|2][ 2][4|2][2]<=Has to lead to
			        		//TODO: othr.Dim[7][4][3][6][9]=>Wi to coord!
							/**-----------------------------------------
							 * 		[2][1][4][6][8][6]
							 * 		[3][-1][2][-4][-5][6]//[0][0][0]=>3 times othr
							 * 		is really:
							 * 		[4  ][-2][1  ][-6][-8][6  ]
							 *		   	   |	    |   |
							 * 		[  6][ 7][  2][ 1][ 4][  8] <= then multiplying with this
							 * 		[4|6][ 6][1|2][ 6][ 5][6|8]
							 *
							 * */
				     		//==================================================
				     		ConvectionStep.on(Vi, Ii, Ni, Ci, Wi);// TODO: Apply with dimensionality considered!
				     		//==================================================
			     			Wi++;
			     			Ci++;
			        	}
				     	Data.addToInput(Vi, Ii, update[0]);
				    }
				}
			}
		}
		shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: return Neuron[Vi].Data;");
		shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: END\n");
		return Data;
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NExecutable
	public NVData activationOf(NVData Data, int Vi)
	{
		shoutLine("NVertex_Root->activationOf(double[][] E.Data, int Vi)|: ...");
		shoutLine("========================================================================");
		if(isParalyzed()==true)
		{
			shoutLine("-activationOf(...,'Vi':"+Vi+")|: (isParalyzed() == true):");
			shoutLine("-activationOf(...,'Vi':"+Vi+")|: => return E.Data;");
			shoutLine("-activationOf(...,'Vi':"+Vi+")|: => END\n");
			return Data;
		}
		NVFunction Function = (NVFunction) findModule(NVFunction.class);
		if(Function!=null)
		{
			shoutLine("-activationOf(...,'Vi':"+Vi+")|: (Function != null):");
			double[] Bias = this.getBias(Vi);
			Data.setActivation(Vi, Function.activate(Data.getInput(Vi), Bias));
			Data.setInputDerivative(Vi, new double[Connection.length]);
			shoutLine("-activationOf(...,'Vi':"+Vi+")|: => for(int Ii=0; Ii<E.Data[2].length; Ii++):");
			for(int Ii=0; Ii<Data.getInputDerivative()[Vi].length; Ii++) // e_get(Vi)!!!!
			{
				shoutLine("-activationOf(...,'Vi':"+Vi+")|: =>('Ii':"+Ii+")|: E.Data[2][Ii]=Function.derive(E.Data[1], Bias, Ii);");
				Data.setInputDerivative(Vi, Ii, Function.derive(Data.getInput(Vi), Bias, Ii));
			}
		}
		else
		{
			int size;
			size = inputSize();
			Data.setActivation(Vi, 1);
			for(int Ii=0; Ii<size; Ii++)
			{
				Data.setActivation(Vi, Data.getActivation(Vi)*Data.getInput(Vi, Ii));
			}
			for(int Ii=0; Ii<size; Ii++)
		    {
				Data.setInput(Vi, Ii, 1);
				for(int i=0; i<size; i++)
				{
					if(Ii!=i) {
						Data.setInputDerivative(
							Vi, Ii,
					      Data.getInputDerivative(Vi, Ii) * Data.getInput(Vi, Ii)
						);
					}
				}
			}
		}
		Data.setError(Vi, 0);
		shoutLine("-activationOf(...,'Vi':"+Vi+")|: return E.Data;");
		shoutLine("-activationOf(...,'Vi':"+Vi+")|: END\n");
		return Data;
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override//NExecutable
	public synchronized boolean memorize(NVData Data)
	{
		shoutLine("NVertex_Root->memorize(double[][][] ResultData)|: ...");
		shoutLine("========================================================================");
		shoutLine("-memorize(...)|: (Property!=null):");
		NVMemory Memory = (NVMemory) findModule(NVMemory.class);
		if(Memory!=null)
		{
			shoutLine("-memorize(...)|: =>(Memory!=null):");
			if(this.weightIsTrainable() || this.biasIsTrainable()) //this can be lambdarized...
			{
				//takeStored = (int Vi)-> {};
			}
			PrimIteration memorize = (int Vi)->{};
			if(State.isParentRoot(this)) // TO DO: -> Checking if size of derivatives array matches memorized arrays (indicates structure change...)
			{
				memorize = (int Vi)->
				{
					double[] relationalDerivative = new double[0];
					shoutLine("-memorize(...)|: =>=>=>('Vi':"+Vi+")|:(State.isParentRoot(this)):");
					NVNode[] tipNodes =
					NVUtility.identityCorrected.order(
						find((NVNode node)->
							{
								if(State.isRootTip(node.asCore()))
								{
									return true;
								}
								return false;
							})
						);
					shoutLine("-memorize(...)|: =>=>=>('Vi':"+Vi+")|:=> for(int Ti=0; Ti<tipNodes.length; Ti++)|: ... ");
					for(int Ti=0; Ti<tipNodes.length; Ti++)
					{
						shoutLine("-memorize(...)|: =>=>=>('Vi':"+Vi+")|:('Ti':"+Ti+")|:=> ...");
						shoutLine("-memorize(...)|: =>=>=>('Vi':"+Vi+")|:('Ti':"+Ti+")|:=> double[] foundrelationalDerivative = findRootrelationalDerivativeOf(tipNodes[Ti], inputRelationalDeriviation, Vi);");
						double[] foundrelationalDerivative = findRootDerivativesOf(tipNodes[Ti], Data, Vi);
						shoutLine("-memorize(...)|: =>=>=>('Vi':"+Vi+")|:('Ti':"+Ti+")|:=> double[] newrelationalDerivative = new double[selfrelationalDerivative[Vi].length + foundrelationalDerivative.length]; ");
						double[] newrelationalDerivative = new double[relationalDerivative.length + foundrelationalDerivative.length];
						shoutLine("-memorize(...)|: =>=>=>('Vi':"+Vi+")|:('Ti':"+Ti+")|:=> for(int Di=0; Di<selfrelationalDerivative[Vi].length; Di++)");
						for(int Di=0; Di<relationalDerivative.length; Di++)
						{
							shoutLine("-memorize(...)|:('Vi':"+Vi+")|:('Ti':"+Ti+")|:('Di':"+Di+")|: =>=>=>=> 'newrelationalDerivative[Di]':"+newrelationalDerivative[Di]+" = 'selfrelationalDerivative[Vi][Di]':"+relationalDerivative[Di]+";");
							newrelationalDerivative[Di]=relationalDerivative[Di];
						}
						shoutLine("-memorize(...)|: =>=>=>('Vi':"+Vi+")|:('Ti':"+Ti+")|:=> for(int Di=0; Di<selfrelationalDerivative[Vi].length; Di++)");
						for(int Di=relationalDerivative.length; Di<relationalDerivative.length+foundrelationalDerivative.length; Di++)
						{
							shoutLine("-memorize(...)|: =>=>=>('Vi':"+Vi+")|:('Ti':"+Ti+")|:('Di':"+Di+")|:=> 'newrelationalDerivative[Di]':"+newrelationalDerivative[Di]+" = 'foundrelationalDerivative[Di-selfrelationalDerivative[Vi].length]':"+foundrelationalDerivative[Di-relationalDerivative.length]+";");
							newrelationalDerivative[Di]=foundrelationalDerivative[Di-relationalDerivative.length];
						}
						shoutLine("-memorize(...)|: =>=>=>('Vi':"+Vi+")|:('Ti':"+Ti+")|:=> selfrelationalDerivative[Vi] = newrelationalDerivative;");
						relationalDerivative = newrelationalDerivative;
					}
					if(relationalDerivative.length==0){relationalDerivative=null;}
					Data.setRelationalDerivative(Vi, relationalDerivative);
				};
			}//---------------------------
			for(int Vi=0; Vi<this.size(); Vi++)
			{
				//===================================
				memorize.on(Vi);
				//===================================
			}
			shoutLine("-memorize(...)|: =>=> Memory.storeCurrentActivity(true);");
			Memory.storeCurrentActivity(true);
			Memory.storeCurrentData(Data);//activity/derivative storing should be deletable
			shoutLine("-memorize(...)|: =>|>");
		}
		//->Local Data gets reinstantiated -> however the array lives on in memory and possibly as double compartment!
		//This procedure is especially important for basic root nodes WITHOUT
		shoutLine("-memorize(...)|: return false;");
		shoutLine("-memorize(...)|: END\n");
		return false;
	}
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override //NExecutable
    public boolean loadLatest()
    {
        return this.loadLatest(BigInteger.ZERO);
    }
    @Override //NExecutable
    public boolean loadLatest(BigInteger Hi){
        //if this is super root -> traverse root preparation!
        if(this.is(NVertex.Child))
        {
            return false;
        }
        return loadState(Hi, false);
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //Backpropagation:
    @Override//NNeuron
    public boolean loadTrainableState(BigInteger Hi)
    {
        return loadState(Hi, true);
    }
	public boolean loadState(BigInteger Hi, boolean preBackprop)
	{
		/*if this is super root node -> test if this is connected to a memory node
		 * -> remove yourself from synchronization object.
		 * -> if counter becomes negative-> adjust size!
		 * */
		shoutLine("NVertex_Root->preRootBackward(BigInteger 'Hi':"+Hi.longValue()+")|: ...");
		shoutLine("========================================================================");
		NVMemory Memory = (NVMemory)findModule(NVMemory.class);
		this.Signal = 1;
		NVData Data = null;
		if (Memory!=null)//this.is(BasicRoot)==false
        {
            int timeDelta = Memory.activeTimeDeltaWithin(Hi);
            Data = Memory.getDataAt(timeDelta);
            this.load(Data);
            long phase = Memory.currentActivityPhase();
            if (Hi.signum() == 1) {
                phase = Memory.activityPhaseAt(Hi);
            }
            this.Signal = phase;
        }
        if (Data == null) {
            Data = this;
        }
        boolean traverse = !(Connection==null||State.isRoot(this)==false);
        if(traverse){
            for (int Ii = 0; Ii < Connection.length; Ii++) {
                if (Connection[Ii] != null) {
                    for (int Ni = 0; Ni < Connection[Ii].length; Ni++) {
                        if (State.isRoot(Connection[Ii][Ni].asCore()) == true) {
                            if (State.isChild(Connection[Ii][Ni].asCore()) == true) {
                                ((NVertex_Root) Connection[Ii][Ni].asCore()).loadState(Hi, preBackprop);
                            }
                        }
                    }
                }
            }
        }
        //----------
        if(this.isDerivable()==false || !preBackprop)
        {
            return false;
        }
        if (Memory!=null)//this.is(BasicRoot)==false
        {
		 	//Self convergent node: Converges without loss node (probably is loss node)
			for(int Vi=0; Vi<this.size(); Vi++)
			{
		 		if (this.isConvergent() && this.isLast()==false)
			 	{//=========================================================
			 		 Neuron[Vi].calculateError();
			 	}//=========================================================
			 	else if(this.isConvergent()==false || this.isLast()==true)
			 	{//=========================================================
			 		int inactive = 1 + Memory.localInactiveTimeDeltaWithin(Hi);
			 		Neuron[Vi].setError(Neuron[Vi].getError()/inactive);//Error has accumulated -> calculating average accordingly!
				}//==========================================================================================
				//Why is that? => Current Error has been calculated
				Data.setError(Vi, this.getError(Vi));//=>Stored in history!
				this.setError(Vi, 0);//Locally e_set to 0 (ready to accept error remotely)
				if(Neuron[Vi].getInputDerivative()==null)
			 	{
			 		shoutLine("-preRootBackward(...)|: (Neuron[Vi].Data[2]==null):");
			 		shoutLine("-preRootBackward(...)|: => return false;");
			 		shoutLine("-preRootBackward(...)|: => END\n");
			 		return false;
			 	}
			 	for (int Di = 0; Di < Neuron[Vi].getInputDerivative().length; Di++)
			 	{
			 		shoutLine("-preRootBackward(...)|:('Di':"+Di+")|: Neuron[Vi].Data[2][Di]:(" + Neuron[Vi].getInputDerivative()[Di] + ");");
			 	}
		 	}
		}
		if(this.ActivitySignalAt(Hi))
		{
			shoutLine("-preRootBackward(...)|: return true;");
			shoutLine("-preRootBackward(...)|: END\n");
			return true;
		}
		shoutLine("-preRootBackward(...)|: return false;");
		shoutLine("-preRootBackward(...)|: END\n");
		return false;
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NRoot
	public void Backward(BigInteger Hi)
	{
		if(this.asRoot().isChild()) {return;}
		rootBackward(Hi);
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void rootBackward(BigInteger Hi)
	{
		shoutLine("NVertex_Root->rootBackward(BigInteger Hi)|: ...");
		shoutLine("========================================================================");
		if (!this.isDerivable())//if(!hasError())
		{
			shoutLine("-rootBackward(...)|: Backpropagation not possible! This neuron is unfit for gradient storing.");
			return;
		}
		if (this.hasWeight(0, 0))
		{
			if(this.hasWeightGradient(0, 0)==false && weightIsTrainable())
			{
				shoutLine("-rootBackward(...)|: instantiateWeightGradient()");
				instantiateWeightGradient();
			}
		}
		if (this.hasBias(0, 0))
		{
			if(this.hasBiasGradient(0, 0)==false && biasIsTrainable())
			{
				shoutLine("-rootBackward(...)|: instantiateShiftGradient()");
			    instantiateBiasGradient();
			}
		}
		NVData[] Data = {null};//this
		NVMemory Memory = (NVMemory)findModule(NVMemory.class);
		if (Memory!=null)//this.is(BasicRoot)==false
		{
			int timeDelta = Memory.activeTimeDeltaWithin(Hi);
			Data[0] = Memory.getDataAt(timeDelta);
		}
		NVOptimizer GradientOptimizer = (NVOptimizer)findModule(NVOptimizer.class);
		//=========================
		//-> Checking if the from memory loaded gradients
		//   match the current (root) node structure
		NVNode[] Tip = NVUtility.identityCorrected.order(
		find((NVNode node)->
		{
			if(State.isRootTip(node.asCore())) {return true;}
			return false;
		}));
		double[] currentError = {0.0};
		int[] deriviationOffset = {0};

		SecoIteration CalculateInputError = (int Vi, int Ii)->{};
		if(weightIsTrainable() || biasIsTrainable())
		{
			CalculateInputError = (int Vi, int Ii)->
			{
				currentError[0] = Data[0].getError(Vi) * this.getInputDerivative(Vi, Ii);
				deriviationOffset[0]++;
			};
		}
		else
		{
			CalculateInputError = (int Vi, int Ii)->
			{
				currentError[0]=0.0;
			};
		}
		QuinIteration CalculateWeightGradient;
 		if (isGradientSumming())
		{
 			CalculateWeightGradient = (int Vi, int Ii, int Ni, int Ei, int Wi)->
 			{
 				if(this.hasWeight(Vi, Ii)==false) {return;}
 				this.addToWeightGradient(Vi, Ii, Wi, Connection[Ii][Ni].getActivation(Ei) * currentError[0]);
 			};
 		}
 		else
 		{
 			CalculateWeightGradient = (int Vi, int Ii, int Ni, int Ei, int Wi)->
 			{
 				if(this.hasWeight(Vi, Ii)==false) {return;}
 				this.setWeightGradient(Vi, Ii, Wi, Connection[Ii][Ni].getActivation(Ei) * currentError[0]);
	 		};
 		}
 		SecoIteration ApplyBiasGradient = (int Vi, int Ii)->{};
		if (isGradientSumming())
		{
			ApplyBiasGradient = (int Vi, int Ii)->
			{
				this.addToBiasGradient(Vi, Ii, currentError[0]);
			};
		}
		else
		{
			if (isInstantGradientApply())
			{
			    if (GradientOptimizer != null)
				{
				    ApplyBiasGradient = (int Vi, int Ii)->
					{
						this.setBiasGradient(Vi, Ii, currentError[0]);
				    	double optimized = GradientOptimizer.getOptimized(this.getBiasGradient(Vi, Ii), Vi, Ii);
				    	this.addToBias(Vi, Ii, optimized);
					};
				}
			    else
			    {
			    	ApplyBiasGradient = (int Vi, int Ii)->
					{
						this.setBiasGradient(Vi, Ii, currentError[0]);
					    this.addToBias(Vi, Ii, this.getBiasGradient(Vi, Ii));
					};
			    }
			}
			else
			{
			    ApplyBiasGradient = (int Vi, int Ii)->
			    {
					this.setBiasGradient(Vi, Ii, currentError[0]);
				};
			}
		}
 		QuinIteration ApplyWeightGradient = (int Vi, int Ii, int Ni, int Ei, int Wi)->{};
 		if(!isGradientSumming())
		{
			if(isInstantGradientApply() && weightIsTrainable())
			{
				if(GradientOptimizer != null)
				{
					ApplyWeightGradient = (int Vi, int Ii, int Ni, int Ei, int Wi)->
					{
						if(this.hasWeight(Vi, Ii)==false) {return;}
						double optimized = GradientOptimizer.getOptimized(this.getWeightGradient(Vi, Ii, Wi),Vi,Ii,Wi);
						this.addToWeight(Vi, Ii, Wi, optimized);
					};
				}
				else
				{
					ApplyWeightGradient = (int Vi, int Ii, int Ni, int Ei, int Wi)->
					{
						if(this.hasWeight(Vi, Ii)==false) {return;}
						this.addToWeight(Vi, Ii, Wi, this.getWeightGradient(Vi, Ii, Wi));
					};
				}
			}
		}
		//=========================
		// Start:
		for(int Vi=0; Vi<this.size(); Vi++)
		{
			deriviationOffset[0] = 0;
			for (int Ii = 0; Ii < Connection.length; Ii++)
			{
				currentError[0] = 0.0;
				CalculateInputError.on(Vi, Ii);
				// Shift/Value Gradient Calculation:
				// ---------------------------------------------------------------------------------------------------
				if (Neuron[Vi].hasBias(Ii) && biasIsTrainable())//  && InputIsSleeping==false // && check == true
				{
					ApplyBiasGradient.on(Vi, Ii);
				}
				// Weight gradient calculation:
				// ---------------------------------------------------------------------------------------------------
				//if (Weight != null || this.isSuperRootNode()) {// && check == true
				//What if this is firstlayer? -> firstlayer neuron doesn't e_get this far! -> (except it is root node (root input node)!)

				if(Connection[Ii]!=null)//&& InputIsSleeping==false//->why not? ->even silent input nodes can receive error values!
				{
					QuinIteration distributeError = (int vi, int ii, int ni, int ei, int wi)->{};
					if(isRootConnection(Connection[Ii])==false)
					{
						distributeError =
						(int vi, int ii, int ni, int ei, int wi)->
						{
							shoutLine("-rootBackward(...)|: ('Vi':"+vi+")|:('Ii':"+ii+")|:=>('Ni':"+ni+")|:('Wi':"+wi+",'Ei':"+ei+")|:=>=>(State.isRoot(Connection[Ii][Ni].Core())):");
							// ================================================================================================================
							if(Neuron[vi].Input[ii].Weight!=null)
							{
								Connection[ii][ni].addToError(ei, currentError[0] * this.getWeight(vi, ii, wi));
							}
							else
							{
								Connection[ii][ni].addToError(ei, currentError[0] * 1);
							}
							// Distributing error to previous layer!
							// ================================================================================================================
						};
					}
					int Wi=0;
					for (int Ni = 0; Ni < Connection[Ii].length; Ni++)
					{
						Connection[Ii][Ni]=Connection[Ii][Ni].asNode();
						int limit = Connection[Ii][Ni].size()+Wi;
						int Ei = 0;
						while(Wi<limit)
			            {
			     			CalculateWeightGradient.on(Vi, Ii, Ni, Ei, Wi);
			     			//TODO: implememt with if(isRootConnection(Node[] con))
			     			distributeError.on(Vi, Ii, Ni, Ei, Wi);
			     			// APPLYING WEIGHT GRADIENTS:
			     			ApplyWeightGradient.on(Vi, Ii, Ni, Ei, Wi);
			     			Wi++;
							Ei++;
			            }//-----------------
					 } // Value neuron loop closed!
				}//Ii connection != null
				// ---------------------------------------------------------------------------------------------------
			} // Value loop closed!
			//GRADIENT DISTRIBUTION:
			// if this is coreheader -> nodes = getRimNodes
			if(this.isFirst()==false)
			{//First layer nodes DO NOT DISTRIBUTE GRADIENTS! (never)
				if(State.isParentRoot(this) || State.isMemoryChild(this))
				{
					Tip = NVUtility.identityCorrected.order(Tip);
					int Ti = 0;

					int Ri = 0;
					double[] relationalDerivative = Neuron[Vi].getRelationalDerivative();//WIP
					while (Ti < Tip.length)
					{
						if(Tip[Ti]!=null)
						{
							if (State.isParentRoot(Tip[Ti].asCore()))
							{//==================================
								Tip[Ti].addToError(Vi, Data[0].getError(Vi) * relationalDerivative[Ri]);//ERROR DISTRIBUTION!
								Ri++;//deriviationOffset[0]++;
							}//==================================
							else if(State.isTrainableBasicChild(Tip[Ti].asCore()))
							{//==================================//ERROR HERE....
								//Tip[Ti].setError(Vi, relationalDerivative[Ri]);
								Ri++;//deriviationOffset[0]++;
								double[] targetedNodeGradient = new double[Tip[Ti].asCore().inputSize()];
								for (int Ci = 0; Ci < Tip[Ti].asCore().inputSize() && Ri<relationalDerivative.length; Ci++)
								{
									targetedNodeGradient[Ci]=relationalDerivative[Ri];
									Ri++;//deriviationOffset[0]++;
								}
								Tip[Ti].asCore().setInputDerivative(Vi, targetedNodeGradient);
								Tip[Ti].asExecutable().Backward(Hi);
							}//==================================
							else if(State.isMemoryChild(Tip[Ti].asCore()))
							{//==================================
								shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>(State.isMemoryChild(Tip[Ti].Core())):");
								//... e_add to error ...
								//find synchronization compartment -> register
								//if i am the last one to synchronize -> finish backpropagation!
								shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>|>");
							}//==================================
						}
						Ti++;
					}
				}
			}
		}//VI loop closed!
		if(this.isFirst()==false && (this.weightIsTrainable() || this.biasIsTrainable()))
		{
			for(int Vi=0; Vi<this.size(); Vi++)
			{
				this.setInput(Vi, null);
				//This is important: why? : => weights changed -> input NEEDS to be recalculated!
			}
		}
		if (isInstantGradientApply() && isGradientSumming()) {
			applyGradients();
		}
		if (isGradientSumming() == false) {
			nullGradients();
		}
		if(this.is(BasicRoot) == false)
		{
			for(int Vi=0; Vi<this.size(); Vi++)
			{
				Data[0].setError(Vi, 0);
			}
		}
		shoutLine("-rootBackward(...)|: END\n");
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//NVNeuron implemention ends here!!!!!
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//===============================================================================================================================================
	/*
	 * tip node identifier: Are root units! What else makes them 'tip' nodes?:
	 *
	 * Either they are 'input units' (root input) Or they are memory dependent (don't have
	 * memory compartment but have trainable weights and/or biases!)... Or they are so
	 * called 'Core Header' -> meaning that they have memory -> so they
	 * automatically copy derivatives with respect to connected nodes...
	 *
	 * Do i want to have root nodes that copy activations? ->YES!
	 * why? ->although it takes up memory the advantage is that core calculations
	 * will not take too long to finish due to the fact that every root node will
	 * activate only once! ->therefore the following functions need to exist!
	 *
	 * ->distributeCoreMemoryAllocation() // before fprop
	 * ->distributeCoreMemoryDeallocation() // after fprop
	 *
	 * Where do lambda expressions come into play? ->makes it possible to create
	 * completely custom functions from scratch for neuron cores! Don't have memory
	 * => Memory node not needed! -> But! Performance gains possible.
	 */
	// called in prepareForBackpropagation() function by SUPER NODE:
	// Function: propagateError(double){ if(CoreisFirst()LAyer){->normal backpropagation
	// needs to start->Backpropdata has been e_set}
	// else{double[] ideriv = findCoreDerivateOf(RimNode[Ni], setting) //==> if
	// ideriv ==null ==> Is memory node! ->
	// ...for each input node! -> RimNode[Ni].setbackPropData(double[] ideriv,
	// error)
	// }
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override// NRoot
	public int rootStructureDepth(int level)
	{
		shoutLine("NVertex_Root->rootDepth(int level)|: ...");
		shoutLine("========================================================================");
		shoutLine(level+"");
		if(level>100) {return level;}
		NVertex Core;
		int current = level;
		int deepest = level;
		for (int Ii = 0; Ii < Connection.length; Ii++)
		{
			if (Connection[Ii] != null)
			{
				if(!State.isParentRoot(Connection[Ii][0].asCore()) && !State.isTrainableNeuron(Connection[Ii][0].asCore()))
				{
					for (int Ni = 0; Ni < Connection[Ii].length; Ni++)
					{
						 Connection[Ii][Ni]=Connection[Ii][Ni].asNode();
						 Core = Connection[Ii][Ni].asCore();
					     if(State.isRoot(Core) && !State.isParentRoot(Core) && Core!=this)
					     {
					    	 current = Core.asRoot().rootStructureDepth(level+1);
					    	 if(current>deepest ) {
					    	 	deepest = current;
					    	 }
					     }
					}
				}
			}
		}
		shoutLine("-rootDepth(int level)|: return 'deepest':"+deepest+";");
		shoutLine("-rootDepth(int level)|: END\n");
		return deepest;
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NRoot
	public void allocateChildMemory()
	{
		// Work in progress:
		// if(connection compartment is instance of storage compartment){return!}else...
		NVertex Core;
		for (int Ii = 0; Ii < Connection.length; Ii++) {
			if (Connection[Ii] != null) {
				if(isRootConnection(Connection[Ii])) {
					for (int Ni = 0; Ni < Connection[Ii].length; Ni++)
					{
						Connection[Ii][Ni]=Connection[Ii][Ni].asNode();
						Core = Connection[Ii][Ni].asCore();
					    if(State.isBasicChild(Core) && !Core.isFirst())
					    {
					    	for(int Vi=0; Vi<this.size(); Vi++)
					    	{
					    		if(!Core.isFirst() || Neuron[Vi].hasInput()==false)
					    		{
					    			this.setInput(Vi, new double[inputSize()]);
					    		}
					    		this.setInputDerivative(Vi, new double[inputSize()]);
					    	}
					    }
					    Core.asRoot().asParent().allocateChildMemory();
					}
				}
			}
		}
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NRoot
	public void freeChildMemory()
	{
		// Work in progress:
		// if(connection compartment is instance of storage compartment){return!}
		NVertex Core;
		for (int Ii = 0; Ii < Connection.length; Ii++) {
			if (Connection[Ii] != null)
			{
				if(isRootConnection(Connection[Ii]))
				{
					for (int Ni = 0; Ni < Connection[Ii].length; Ni++)
					{
						Connection[Ii][Ni]=Connection[Ii][Ni].asNode();
						Core = Connection[Ii][Ni].asCore();
						if(State.isBasicChild(Core))
						{
							for(int Vi=0; Vi<this.size(); Vi++)
					    	{
								if(!Core.isFirst())
								{
									Neuron[Vi].setInput(null);
								}
								Neuron[Vi].setInputDerivative(null);
					    	}
						}
						Core.asRoot().asParent().freeChildMemory();
					}
				}
			}
		}
	}
	// GETTING DERIVATIVE OF A SPECIFIC NODE //... coreTraverse?
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NRoot //TODO implement relational derivatives as its own value field!(within element)
	public synchronized double[] findRootDerivativesOf(NVNode connectionNode, NVData Data, int Vi)
	{
		// Finding derivative with respect to a specific neuron
		shoutLine("NVertex_Root->findRootDerivativesOf(NVNode connectionNode, boolean inputDerivative, int Vi)|: ...");
		shoutLine("========================================================================");
		boolean inputDerivative = true;
		if(State.isMemoryChild(connectionNode.asCore()) || State.isParentRoot(connectionNode.asCore()))
		{
			inputDerivative = false;
		}
		if(Data==null){
			Data = this;
		}
		double[] resultDerivatives = null;
		boolean thisIsTargNode = false;
		if (connectionNode.asNode() == this.asNode()) {
			thisIsTargNode=true;
		}
		shoutLine("-findRootDerivativesOf(...)|:|>");
		if (thisIsTargNode)//Target node found: -> returning derivative!
		{
			shoutLine("-findRootDerivativesOf(...)|:(thisIsTargNode):");
			if (inputDerivative==false) {//Derivative with respect to activation. (memory nodes)
				resultDerivatives    = new double[1];
				resultDerivatives[0] = 1;
			}
			else {//Derivative with respect to inputs.
				resultDerivatives = new double[Neuron[Vi].getInputDerivative().length];
				for(int Di = 0; Di < resultDerivatives.length; Di++)
				{
					resultDerivatives[Di] = Neuron[Vi].getInputDerivative(Di);
				}
			}
			shoutLine("-findRootDerivativesOf(...)|:=> return resultDerivatives;");
			shoutLine("-findRootDerivativesOf(...)|:=> END\n");
			return resultDerivatives;
		}
		//if(this.isInputRootNode()) {return null;}
		// why is the above not used? -> input root nodes are now not that obvious anymore -> input root define themselves
		// by havin AT LEAST ONE reference to the outside!
		if (inputDerivative==false) {//Why is that? -> super root stores the relational gradient!
			resultDerivatives = new double[1];
		}
		else {
			resultDerivatives = new double[connectionNode.asCore().inputSize()];
		}
		for(int Ii = 0; Ii < Connection.length; Ii++)
		{
			if (Connection[Ii] != null)
			{
				if(State.isParentRoot(Connection[Ii][0].asCore())==false && State.isRoot(Connection[Ii][0].asCore()))
				{
					double[] inputSumDerivatives = new double[resultDerivatives.length];
					for(int Ni = 0; Ni < Connection[Ii].length; Ni++)
					{
						if (Connection[Ii][Ni] != null)
						{
							double[] rootRelationalDerivatives = null;
							if(Connection[Ii][Ni]!=this && Connection[Ii][Ni]!=this.asNode())
							{
								rootRelationalDerivatives =
										Connection[Ii][Ni].
											asCore().
											asRoot().
											findRootDerivativesOf(connectionNode, null, Vi);
							}
							if(rootRelationalDerivatives != null)
							{
								for(int Di = 0; Di < rootRelationalDerivatives.length; Di++)
								{
									if(Neuron[Vi].Input[Ii].Weight != null) {
										rootRelationalDerivatives[Di] *= Neuron[Vi].Input[Ii].Weight[Ni];
										//Why not Wi? -> 1:1 relations within root structure!
									}
									inputSumDerivatives[Di] += rootRelationalDerivatives[Di];
								}
							}
						}
					}
					for(int Di = 0; Di < resultDerivatives.length; Di++)
					{
						if(Data.hasInputDerivative(Vi)){
							inputSumDerivatives[Di] *= Data.getInputDerivative(Vi, Ii);
						}
						resultDerivatives[Di] += inputSumDerivatives[Di];
					}
					// =============
				}
			}
		}// Ii loop closed
		shoutLine("-findRootDerivativesOf(...)|: return resultDerivatives;");
		shoutLine("-findRootDerivativesOf(...)|: END\n");
		return resultDerivatives;
	}
	/*
	 * What is the use case of the above function?
	 *
	 * The function is not in use as of yet. It allows one to e_get the derivative of
	 * an addressed node in a directed nodegraph. A recursive graph structure will break this function.
	 * I'm planning on using it as a
	 * tool to collect derivatives of so called "e_sub neurons". These e_sub-Neurons are
	 * components of a neurons Calculation Compartment. They form a node graph
	 * capable of forming any neuron function. The purpose is full customizability
	 * of neuron cores! Lambda expressions will be implemented!
	 */
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NRoot
	public void distributeRootRevival()// Needs to be called by the output core-node of the entire super
	{									// Core! -> otherwise no new activation will take place!
		shoutLine("NVertex_Root->distributeRootRevival()|: ...");
		shoutLine("========================================================================");
		for (int Ii = 0; Ii < Connection.length; Ii++)
		{
			if (Connection[Ii] != null)
			{
				if(State.isParentRoot(Connection[Ii][0].asCore())==false && State.isRoot(Connection[Ii][0].asCore())==true)
				{
					setParalyzed(false);
					if(Connection[Ii]!=null)
					{
						for(int Ni = 0; Ni < Connection[Ii].length; Ni++)
						{
							Connection[Ii][Ni].asRoot().asParent().distributeRootRevival();
						}
					}
				}
			}
		}
	}
	// NOTE: This function will only forward new output if recalculation permission
	// has been propagated or the connection nodes are have memory!
	// ALSO: This function will not act recursively if the connection node is not of
	// type core! -> these functions serve the purpose of being core node functions
	// exclusively!
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NRoot
	public double findRootActivation(int Vi)
	{
		shoutLine("NVertex_Root->findRootActivation(int Vi)|: ...");
		shoutLine("========================================================================");
		//Higher level nodes:
		if(this.is(NVertex.Root)==false || this.is(NVertex.MotherRoot) || this.is(NVertex.RootMemory))
		{
			shoutLine("-findRootActivation(...)|: (this.is(NVertex.Root)==false || this.is(NVertex.MotherRoot) || this.is(NVertex.RootMemory)):");
			shoutLine("-findRootActivation(...)|: => return Activation(Vi);");
			return getActivation(Vi);
		}
		//========================================
		//Basic:
		if(isParalyzed()==true)
		{
			shoutLine("(isParalyzed()==true)!!!");
		}
		if (Neuron[Vi].getInput()!=null && isParalyzed()==false) //This is not entered wehen vi==1
		{
			boolean[] Signal;
			if(this.is(NVertex.RootInput)) {
				Signal = this.inputSignalIf(true);
			}
			else
			{
				if(Neuron[Vi].hasInput()==false)// This might not be right....
				{
					Signal = new boolean[inputSize()];// This node has been active! (weight convection)
					for(int Ii=0; Ii<Signal.length; Ii++)
					{
						Signal[Ii]=false;
					}
				}
				else
				{
					Signal = new boolean[inputSize()];//This node has not been active!
					for(int Ii=0; Ii<Signal.length; Ii++)
					{
						Signal[Ii]=true;
					}
				}
			}
			NVData Data = new NVData(Neuron);
			Data =
			activationOf(
					weightedConvectionOf(
							Data,
							publicized(Signal),
							Vi),
					Vi
			);//publicized is not needed!
			this.Neuron = Data.getDataAsElementArray();
		}
		shoutLine("-findRootActivation(...)|: => return Neuron[Vi].Output()[0];");
		return Neuron[Vi].getActivation();
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	/* The following method traverses the root structure
	 * in order to activate or deactivate nodes based on input activity.
	 * the phaseSignal is provided by the mother...
	 * */
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NVRoot
	public boolean isParent()
	{
		if(Parent==null || Parent==this) {
			return true;
		}
		return false;
	}
	@Override //NVRoot
	public NVParent asParent()
	{
		if(Parent==null || Parent==this) {
			return this;
		}
		return null;
	}
	@Override //NVRoot
	public boolean isChild()
	{
		if(Parent==null || Parent==this) {
			return false;
		}
		return true;
	}
	@Override //NVRoot
	public NVChild asChild()
	{
		if(Parent==null || Parent==this) {
			return null;
		}
		return (NVChild)this;
	}
	@Override //NVChild
	public NVParent getParent()
	{
		return (NVParent)Parent;
	}
	@Override //NVChild
	public void setParent(NVParent mother)
	{
		if(((NVertex)mother).size()==this.size()) {
			Parent = (NVRoot)mother;
		}
	}
	public long getActivitySignal()
	{
		if(this.isFirst()){
			return 1;
		}
		return Signal;
	}

}//Class end.
//=========================================================================================================
