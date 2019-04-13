package napi.main.core.imp;

import java.math.BigInteger;

import napi.main.core.NVChild;
import napi.main.core.NVNeuron;
import napi.main.core.NVNode;
import napi.main.core.NVParent;
import napi.main.core.NVRoot;
import napi.main.core.NVUtility;
import napi.main.core.NVertex;
import napi.main.core.calc.NVFunction;
import napi.main.core.comp.NVData;
import napi.main.core.mem.NVMemory;
import napi.main.core.opti.NVOptimizer;
import napi.main.exec.NVExecutable;
import ngui.NPanelNode;

public class NVertex_Root extends NVertex implements NVNeuron, NVRoot, NVExecutable, NVChild, NVParent
{
		private NVRoot Parent;
		private long Signal;
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		public NVertex_Root(int size, boolean isFirst, boolean isLast, int neuronID, int functionID) 
		{
			super(size, isFirst, isLast, neuronID, functionID);
			shoutLine("new NVertex_Root(int size="+size+", boolean isFirst="+isFirst+", boolean isLast="+isLast+", int v_id="+neuronID+", int f_id="+functionID+")|: ...created!");
			//addModule(new NVCloak(this));
			setParent((NVParent)this);
		}
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		public NVertex_Root(int size, boolean isFirst, boolean isLast, int neuronID, int functionID, int tunnel) 
		{
			super(size, isFirst, isLast, neuronID, functionID, tunnel);
			shoutLine("new NVertex_Root(int size="+size+", boolean isFirst="+isFirst+", boolean isLast="+isLast+", int v_id="+neuronID+", int f_id="+functionID+", int delay="+tunnel+")|: ...created!");
			//addModule(new NVCloak(this));
			setParent((NVParent)this);
		}
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		public NVertex_Root(NVNode parent, boolean isFirst, boolean isLast, int neuronID, int functionID) 
		{
			super(parent, isFirst, isLast, neuronID, functionID, 1);
			shoutLine("new NVertex_Root(NVNode mother=#"+parent.asCore().getID()+", boolean isFirst="+isFirst+", boolean isLast="+isLast+", int v_id="+neuronID+", int f_id="+functionID+")|: ...created!");
			if(parent==null) 
			{
				//addModule(new NVCloak(this));
				setParent((NVParent)this);
			}
			else
		    {
				setParent(parent.asRoot().asParent());
		    }
		}
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		public NVertex_Root(NVNode parent, boolean isFirst, boolean isLast, int neuronID, int functionID, int tunnel) 
		{
			super(parent, isFirst, isLast, neuronID, functionID, tunnel);
			shoutLine("new NVertex_Root(NVNode mother=#"+parent.asCore().getID()+", boolean isFirst="+isFirst+", boolean isLast="+isLast+", int v_id="+neuronID+", int f_id="+functionID+", int delay="+tunnel+")|: ...created!");
			if(parent==null) 
			{
				//addModule(new NVCloak(this));
				setParent((NVParent)this);
			}
			else
		    {
				setParent(parent.asRoot().asParent());
		    }
		}
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		public NVertex_Root(NVertex toBeCopied)
		{
			super(toBeCopied);
			//addModule(new NVCloak(this));
			setParent(this);
		}//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		@Override //NExecutable
		public boolean loadState()
		{
			return this.loadState(BigInteger.ZERO);
		}
		@Override //NExecutable
		public boolean loadState(BigInteger Hi){
			//if this is super root -> traverse root preparation!
			if(this.is(NVertex.Child))
			{
				return false;
			}
			return cleanupChildren(Hi);
		}
		// Forward/Backward Propagation Functions:
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
		private boolean cleanupChildren(BigInteger Hi)
		{
			shoutLine("NVertex_Root->cleanupChildren()|: ...");
			shoutLine("========================================================================");
			NVMemory Memory = (NVMemory)findModule(NVMemory.class);
			if(Memory==null) 
			{
				this.Signal = 1;
				return true;
			}
			int activityDelta = Memory.activeTimeDeltaWithin(Hi);
			this.load(Memory.getDataAt(activityDelta));
			long phase = Memory.currentActivityPhase();
			if(Hi.signum()==1){phase = Memory.activityPhaseAt(Hi);}
			this.Signal = phase;

			if(State.isRoot(this) == false)
			{
				 shoutLine("-cleanupChildren()|: (this == root): => return false; (Not within root anymore...) ");
				 shoutLine("-cleanupChildren()|: => END\n");
				 return false;
			}// Outside of root!
			if(Connection==null)
			{
				 shoutLine("-cleanupChildren()|: (Connection == null): => return false; (deadend...) ");
				 shoutLine("-cleanupChildren()|: => END\n");
				 return false;
			}
			//---------------------------------------
			shoutLine("-cleanupChildren()|: for(int Ii=0; Ii<Connection.length; Ii++):");
			for(int Ii=0; Ii<Connection.length; Ii++)
			{
				shoutLine("-cleanupChildren()|:('Ii':"+Ii+")|:  ");
				if(Connection[Ii]!=null)
				{
					shoutLine("-cleanupChildren()|:('Ii':"+Ii+")|: (Connection["+Ii+"] != null):");
					shoutLine("-cleanupChildren()|:('Ii':"+Ii+")|: => for(int Ni=0; Ni<Connection[Ii].length; Ni++)");
					for(int Ni=0; Ni<Connection[Ii].length; Ni++)
					{
						shoutLine("-cleanupChildren()|:('Ii':"+Ii+")|:('Ni':"+Ni+")|:  ");
						if(State.isRoot(Connection[Ii][Ni].asCore())==true)
						{
							shoutLine("-cleanupChildren()|:('Ii':"+Ii+")|:('Ni':"+Ni+")|: (Connection["+Ii+"]["+Ni+"].Core == root):");
							shoutLine("-cleanupChildren()|:('Ii':"+Ii+")|:('Ni':"+Ni+")|: =>Connection["+Ii+"]["+Ni+"] = Connection["+Ii+"]["+Ni+"].Node()  ");
							Connection[Ii][Ni] = Connection[Ii][Ni].asNode();
							//-----------------------------------------------
							if(State.isChild(Connection[Ii][Ni].asCore()) == true)
							{
								shoutLine("-cleanupChildren()|:('Ii':"+Ii+")|:('Ni':"+Ni+")|: (Connection["+Ii+"]["+Ni+"].Core == child):");
								shoutLine("-cleanupChildren()|:('Ii':"+Ii+")|:('Ni':"+Ni+")|: => Connection["+Ii+"]["+Ni+"].Core().Root().cleanupChildren();  ");
								((NVertex_Root)Connection[Ii][Ni].asCore()).cleanupChildren(Hi);
							}
						}
					}
				}
			}
			shoutLine("-cleanupChildren()|: Current activity phase: "+phase);
			if(phase>0) 
			{
				shoutLine("-cleanupChildren()|: (phase > 0): => return true;");
				shoutLine("-cleanupChildren()|: => END\n");
				return true;
			}
			else
			{
				shoutLine("-cleanupChildren()|: (phase < 0): => return false;");
				shoutLine("-cleanupChildren()|: => END\n");
				return false;
			}
		}
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		@Override //NExecutable
		public void forward() 
		{
			shoutLine("NVertex_Root->forward()|: ...");
			shoutLine("========================================================================");
			//if this is sub root return else startRootForward!
			if(this.is(NVertex.Child)) 
			{
				shoutLine("-forward()|: (this == child): => return");
				shoutLine("-forward()|: END\n");
				return;
			}
			//result should be NVElement[]!
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
					weightedConvectionOf(signal, Vi), 
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
	 			shoutLine("-forwardCondition()|: This is dependend sub-root node. Node cannot perform forward propagation on its own.");
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
	 						boolean childSignal = // Correct to distributeInputSignal
	 								Connection[Ii][Ni].asRoot().distributeRootSignal(selfState);
	 						
	 						if(childSignal==true)
	 						{
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
				if (PanelNode != null) 
				{
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
				if(Connection[0].is(NVertex.Child)) 
				{
					return true;
				}
			}
			return false;
		}
		@Override	
		public NVData weightedConvectionOf(boolean[] Signal, int Vi) 
		{
			shoutLine("NVertex_Root->startWeightConvection(boolean[] inputSignal, int Vi)|: ...");
			shoutLine("========================================================================");
			if(Element[Vi].hasInput()==false) {Element[Vi].setInput(new double[Connection.length]);}//maybe return instead?
			/*Why would LD[1] (input values) ever be null?!?!
			 *=> Some root units (basic root type) set their input arrays to null in order to safe memory.
			 * */
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
			QuinIteration ConvectionStep = null;
 			ConvectionStep = 
 			(int vi, int ii, int ni, int ei, int wi)->
			{
				shoutLine("-weightConvectionOf(...,'vi':"+vi+")|: ('Ii':"+ii+")|:=>('Ni':"+ni+")|:('Wi':"+wi+", 'Ci':"+ei+")|:(State.isRoot(this) == true):");
	     		if(Connection[ii][ni] != Connection[ii][ni].asNode()) 
	     		{
	     			Connection[ii][ni] = Connection[ii][ni].asNode();
	     		}
	     		shoutLine("-weightConvectionOf(...,'vi':"+vi+")|: ('Ii':"+ii+")|:=>('Ni':"+ni+")|:('Wi':"+wi+", 'Ci':"+ei+")|:=>(Element[vi].Dendrite[Ii].Weight != null):");
	     		if(Connection[ii][ni]!=asNode() && isRootConnection[0]) 
	     		{
	     			shoutLine("-weightConvectionOf(...,'vi':"+vi+")|: ('Ii':"+ii+")|:=>('Ni':"+ni+")|:('Wi':"+wi+", 'Ci':"+ei+")|:=>=>(Connection[Ii][Ni]!=Node() && isRootConnection[0]):");
	     			if (Element[vi].Variable[ii].Weight == null) // WHY? => inner root nodes dont need weighted connections
	     			{
	     			  	shoutLine("-weightConvectionOf(...,'vi':"+vi+")|: ('Ii':"+ii+")|:=>('Ni':"+ni+")|:('Wi':"+wi+", 'Ci':"+ei+")|:=>=>=>(Element[vi].Dendrite[Ii].Weight == null):");
	     			  	shoutLine("-weightConvectionOf(...,'vi':"+vi+")|: ('Ii':" + ii + ")|:=>('Ni':" + ni + ")|:('Wi':"+wi+", 'Ci':"+ei+")|:=>=>=>=> 'update[0]':"+update[0]+" = Connection[Ii][Ni].Root().findRootActivation(vi) * 1;");
	     			  	update[0] += Connection[ii][ni].asRoot().findRootActivation(vi) * 1;
	     			  	shoutLine("-weightConvectionOf(...,'vi':"+vi+")|: ('Ii':" + ii + ")|:=>('Ni':" + ni + ")|:('Wi':"+wi+", 'Ci':"+ei+")|:=>=>=>=> 'update[0]':"+update[0]+"");
	     			} 
	     			else //If connection == this ==> bn
	       		   	{
	     			  	shoutLine("-weightConvectionOf(...,'vi':"+vi+")|: ('Ii':"+ii+")|:=>('Ni':"+ni+")|:('Wi':"+wi+", 'Ci':"+ei+")|:=>=>=>(Element[vi].Dendrite[Ii].Weight != null):");
	     			  	update[0] += Connection[ii][ni].asRoot().findRootActivation(vi) * Element[vi].Variable[ii].Weight[wi];
	     			  	shoutLine("-weightConvectionOf(...,'vi':"+vi+")|: ('Ii':" + ii + ")|:=>('Ni':" + ni + ")|:('Wi':"+wi+", 'Ci':"+ei+")|:=>=>=>=> update[0] = Connection[Ii][Ni].Root().findRootActivation(vi) * Element[vi].Dendrite[Ii].Weight[Wi];");
	     			}
	     		}
	     		else 
	    		{
	   				shoutLine("-weightConvectionOf(...,'vi':"+vi+")|: ('Ii':"+ii+")|:=>('Ni':"+ni+")|:('Wi':"+wi+", 'Ci':"+ei+")|:=>=>((Connection[Ii][Ni]!=Node() && isRootConnection[0])==false):");
	    			if(Element[vi].Variable[ii].Weight == null) // WHY? :=> inner root nodes don't need weighted connections
	     			{
	     			  	shoutLine("-weightConvectionOf(...,'vi':"+vi+")|: ('Ii':"+ii+")|:=>('Ni':"+ni+")|:('Wi':"+wi+", 'Ci':"+ei+")|:=>=>=>(Element[vi].Dendrite[Ii].Weight == null):");
	     			  	update[0] += (Connection[ii][ni].getActivation(ei)-Connection[ii][ni].getOptimum(ei)) * 1;
	     			  	shoutLine("-weightConvectionOf(...,'vi':"+vi+")|: ('Ii':" + ii + ")|:=>('Ni':" + ni + ")|:('Wi':"+wi+", 'Ci':"+ei+")|:=>=>=>=> update[0] = Connection[Ii][Ni].Activation(Ci)-Connection[Ii][Ni].Optimum(Ci) * 1;");
	     			} 
	     			else //If connection == this ==> no need for recursive call
	     			{
	     			  	shoutLine("-weightConvectionOf(...,'vi':"+vi+")|: ('Ii':"+ii+")|:=>('Ni':"+ni+")|:('Wi':"+wi+", 'Ci':"+ei+")|:=>=>=>(Element[vi].Dendrite[Ii].Weight != null):");
	     			  	update[0] += (Connection[ii][ni].getActivation(ei)-Connection[ii][ni].getOptimum(ei)) * Element[vi].Variable[ii].Weight[wi];
	     			  	shoutLine("-weightConvectionOf(...,'vi':"+vi+")|: ('Ii':" + ii + ")|:=>('Ni':" + ni + ")|:('Wi':"+wi+", 'Ci':"+ei+")|:=>=>=>=> update[0] = Connection[Ii][Ni].Activation(Ci)-Connection[Ii][Ni].Optimum(Ci) * 'Element[vi].Dendrite[Ii].Weight[Wi]':"+Element[vi].Variable[ii].Weight[wi]+";");
	     			}
	     		}	
			};
 			//===============================================================================
			for (int Ii = 0; Ii < Connection.length; Ii++) 
			{
				shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|: ...");
				if(isRootConnection(Connection[Ii])==false) 
				{
					shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|: if(isRootConnection[0](Connection[Ii])==false):");
					shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=> check =  'Signal[Ii]':"+Signal[Ii]+";");	
				}
				//Note: input signal function must be modified to match this here !!
				if(Signal[Ii]==true && Connection[Ii]!=null) 
				{//Note: -> if an input does not recieve a signal-> the last input will be retained! (resetting it would distort reality)
					shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|: (Signal[Ii]==true && Connection[Ii]!=null):");
					shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=> Element[Vi].Input()[Ii] = 0;");
					//this.setInput(Vi, Ii, 0);
					Element[Vi].setInput(Ii, 0.0);//New Value will be stored here -> therefore: reset to 0
					shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=> double update[0] = 0;");
					
					shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=> boolean isRootConnection[0]=false;");
					isRootConnection[0]=false;
					if(State.isChild(Connection[Ii][0].asNode())) 
					{
						shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=>(State.isChild(Connection[Ii][0].Node())):");
						shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=>=> isRootConnection[0]=true;");
						isRootConnection[0]=true;
					}
					shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':" + Ii + ")|:=> 'isRootConnection[0]':"+isRootConnection[0]);
					shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':" + Ii + ")|:=> int Wi=0;");
				    int Wi=0;
				    shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':" + Ii + ")|:=> for (int Ni = 0; Ni < Connection[Ii].length; Ni++):");
					if(isRootConnection[0])
					{
						for (int Ni = 0; Ni < Connection[Ii].length; Ni++) 
					    {   
							update[0] = 0;
							shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+")|: Connection[Ii][Ni]=Connection[Ii][Ni].Node();");
						    Connection[Ii][Ni]=Connection[Ii][Ni].asNode();
					     	shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+")|: int limit = 'Connection[Ii][Ni].size()':"+Connection[Ii][Ni].size()+" + 'Wi':"+Wi+"; ");
					     	shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+")|: int Ci=0;");
					     	
					     	shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|: ConvectionStep.foreach(Vi, Ii, Ni, Ni, Ni); ..."); 
					     	//==================================================
					     	ConvectionStep.on(Vi, Ii, Ni, Ni, Ni);
					     	//==================================================
					     	shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+")|: 'Element[Vi].Input()[Ii]':"+Element[Vi].getInput()[Ii]+" += 'update[0]':"+update[0]+";");
				     		Element[Vi].addToInput(Ii, update[0]);
					    }//Ni loop closed
					}
					else
					{
						for (int Ni = 0; Ni < Connection[Ii].length; Ni++) 
					    {   
							update[0] = 0;
							shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+")|: Connection[Ii][Ni]=Connection[Ii][Ni].Node();");
						    Connection[Ii][Ni]=Connection[Ii][Ni].asNode();
					     	shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+")|: int limit = 'Connection[Ii][Ni].size()':"+Connection[Ii][Ni].size()+" + 'Wi':"+Wi+"; ");
					     	shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+")|: int Ci=0;");
					     	int limit = Connection[Ii][Ni].size()+Wi; 
					     	int Ci=0;
					     	while(Wi<limit)
				        	{
					     		shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",  'Ci':"+Ci+")|: ..."); 
					     		//==================================================
					     		ConvectionStep.on(Vi, Ii, Ni, Ci, Wi);
					     		//==================================================
					     		shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",  'Ci':"+Ci+")|: Wi++; Ci++;"); 
				     			Wi++; 
				     			Ci++;
				        	}//Wi loop closed
					     	shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+")|: 'Element[Vi].Input()[Ii]':"+Element[Vi].getInput()[Ii]+" += 'update[0]':"+update[0]+";");
				     		Element[Vi].addToInput(Ii, update[0]);
					    }//Ni loop closed	
					}
				}
				shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: ('Ii':" + Ii + ")|: 'Element['Vi':"+Vi+"].Input()['Ii':"+Ii+"]':" + Element[Vi].getInput()[Ii] + "; ");
			}
			shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: return Element[Vi].Data;");
			shoutLine("-weightConvectionOf(...,'Vi':"+Vi+")|: END\n");
			NVData Data = new NVData(Element);
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
				for(int Ii=0; Ii<Data.getInputDerivative()[Vi].length; Ii++) // get(Vi)!!!!
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
					//LocalE.Data[1]*=LocalE.Data[Ii+2];
				}
				for(int Ii=0; Ii<size; Ii++) 
			    {
					Data.setInput(Vi, Ii, 1);
					for(int i=0; i<size; i++) 
					{
						if(Ii!=i) 
						{
							Data.setInputDerivative(Vi, Ii,
									Data.getInputDerivative(Vi, Ii)*Data.getInput(Vi, Ii)
							);
							//E.Data[2][Ii] *= E.Data[1][i];
						}
					}
				}
			}
			shoutLine("-activationOf(...,'Vi':"+Vi+")|: 'E.Data[0][0]':"+Data.getActivation(Vi) + ";");
			shoutLine("-activationOf(...,'Vi':"+Vi+")|: 'E.Data[0][1]':"+Data.getError(Vi)+" = 0;");
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
				//PrimIteration takeStored = (int Vi)->{};
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
							boolean inputRelationalDeriviation = true;
							if(State.isMemoryChild(tipNodes[Ti].asCore()) || State.isParentRoot(tipNodes[Ti].asCore()))
							{
								inputRelationalDeriviation = false;
							}
							shoutLine("-memorize(...)|: =>=>=>('Vi':"+Vi+")|:('Ti':"+Ti+")|:=> double[] foundrelationalDerivative = findRootrelationalDerivativeOf(tipNodes[Ti], inputRelationalDeriviation, Vi);");
							double[] foundrelationalDerivative = findRootDerivativesOf(tipNodes[Ti], inputRelationalDeriviation, Vi);
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
		//Backpropagation:
		@Override//NNeuron
		public boolean preBackward(BigInteger Hi) 
		{
			/*if this is super root node -> test if this is connected to a memory node
			 * -> remove yourself from synchronization object.
			 * -> if counter becomes negative-> adjust size!
			 * */
			shoutLine("NVertex_Root->preRootBackward(BigInteger 'Hi':"+Hi.longValue()+")|: ...");
			shoutLine("========================================================================");
			if(this.isDerivable()==false) 
			{
				shoutLine("-preRootBackward(...)|: Neuron not derivable!");
				return false;
			}
			shoutLine("-preRootBackward(...)|: PREPARING BACKPROPAGATION AT TIME STEP: -" + Hi);
			boolean Signal = ActivitySignalAt(Hi);
			//NEW:
			if(Signal == false)
			{
				this.Signal = -1;
			}
			else
			{
				this.Signal = 1;
			}
			if(Signal == false) 
			{
			 	shoutLine("-preRootBackward(...)|: (Signal == false) => return false; ");
			 	shoutLine("-preRootBackward(...)|: END\n");
			 	return false;
			}
			shoutLine("-preRootBackward(...)|: Loading stored backpropdata: inputderivatives and newly recieved error.",true);
			NVMemory Memory = (NVMemory)findModule(NVMemory.class);
			if (Memory!=null)//this.is(BasicRoot)==false
			{
			 	int timeDelta = Memory.activeTimeDeltaWithin(Hi);
				NVData Data = Memory.getDataAt(timeDelta);
				this.load(Data);
				if(Data==null)
				{
					Data = this;
				}
			 	//Self convergent node: Converges without loss node (probably is loss node)
				for(int Vi=0; Vi<this.size(); Vi++)
				{
			 		shoutLine("-preRootBackward(...)|: =>=>('Vi':"+Vi+")|: Element[Vi].Data[2] = derivatives[Vi];");
			 		//-----------------------------------
			 		if (this.isConvergent() && this.isLast()==false)
				 	{//=========================================================
				 		 shoutLine("-preRootBackward(...)|: ('Vi':"+Vi+")|: Element[Vi].calculateError();");
				 		 //Doesn't consider received error...	
				 		 Element[Vi].calculateError();
				 	}//=========================================================
				 	else if(this.isConvergent()==false || this.isLast()==true)
				 	{//=========================================================
				 		int inactive = 1 + Memory.localInactiveTimeDeltaWithin(Hi);
				 		shoutLine("-preRootBackward(...)|: Inactive time delta: "+inactive);
				 		//==========================================================================================
				 		shoutLine("fetched Error: "+Element[Vi].getError()+"; Memory stored Error: "+Element[Vi].getError());
				 		Element[Vi].setError(Element[Vi].getError()/inactive);//Error has accumulated -> calculating average accordingly!
				 		shoutLine("Time corrected Error: "+Element[Vi].getError()+"; (divided by: "+inactive+") ");
				 		//==========================================================================================
				 	}
					//Why is that? => Current Error has been calculated
					Data.setError(Vi, this.getError(Vi));//=>Stored in history!
					this.setError(Vi, 0);//Locally set to 0 (ready to accept error remotely)

				 	shoutLine("-preRootBackward(...)|: Error:(" + Element[Vi].getError() + "), Element[Vi].Data.length:("+Element[Vi].getIO().length+");");
				 	if(Element[Vi].getInputDerivative()==null)
				 	{
				 		shoutLine("-preRootBackward(...)|: (Element[Vi].Data[2]==null):");
				 		shoutLine("-preRootBackward(...)|: => return false;");
				 		shoutLine("-preRootBackward(...)|: => END\n");
				 		return false;
				 	}
				 	for (int Di = 0; Di < Element[Vi].getInputDerivative().length; Di++) 
				 	{
				 		shoutLine("-preRootBackward(...)|:('Di':"+Di+")|: Element[Vi].Data[2][Di]:(" + Element[Vi].getInputDerivative()[Di] + ");");
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
			if (Element[0].getIO().length<=2)//if(!hasError())
			{
				shoutLine("-rootBackward(...)|: Backpropagation not possible! This neuron is unfit for gradient storing.");
				return;
			}
			if (this.hasInputDerivative(0)==false) 
			{
				shoutLine("-rootBackward(...)|: Backpropagation not possible! LocalData[2]==null");
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
					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:(isWeightTrainable() || isShiftTrainable()):");
					currentError[0] = Data[0].getError(Vi) * this.getInputDerivative(Vi, Ii);
					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=> 'currentError[0]':" + currentError[0] + " = 'Element[Vi].Data[2][Ii]':"+ Element[Vi].getInputDerivative()[deriviationOffset[0]] + " * 'Element[Vi].Output()[1]':" + Element[Vi].getError() + ";");
					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=> 'deriviationOffset[0]':"+deriviationOffset[0]+"++;");
					deriviationOffset[0]++;
				};
			}
			else
			{
				CalculateInputError = (int Vi, int Ii)->
				{
					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:((isWeightTrainable() || isShiftTrainable()) == false):");
					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=> 'currentError[0]':"+currentError[0] +" = 0;");
					currentError[0]=0.0;
				};
			}
			QuinIteration CalculateWeightGradient;
 			if (isGradientSumming()) 
			{
 				CalculateWeightGradient = (int Vi, int Ii, int Ni, int Ei, int Wi)->
 				{
 					if(this.hasWeight(Vi, Ii)==false) {return;}
 					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:(weightsAreSet):");
 					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:(isGradientSumming()):");
 					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:=> 'Element[Vi].Dendrite[Ii].WeightGradient[Wi]':"+Element[Vi].Variable[Ii].WeightGradient[Wi]+" += 'Connection[Ii][Ni].Activation(Ei)':"+Connection[Ii][Ni].getActivation(Ei)+" * 'currentError[0]':"+currentError[0]+";");
 					this.addToWeightGradient(Vi, Ii, Wi, Connection[Ii][Ni].getActivation(Ei) * currentError[0]);
 					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:|>");
 				};
 			} 
 			else
 			{
 				CalculateWeightGradient = (int Vi, int Ii, int Ni, int Ei, int Wi)->
 				{
 					if(this.hasWeight(Vi, Ii)==false) {return;}
 					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:(weightsAreSet):");
	 				shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:(isGradientSumming()==false):");
	 				shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:=> 'Element[Vi].Dendrite[Ii].WeightGradient[Wi]':"+Element[Vi].Variable[Ii].WeightGradient[Wi]+" = 'Connection[Ii][Ni].Activation(Ei)':"+Connection[Ii][Ni].getActivation(Ei)+" * 'currentError[0]':"+currentError[0]+";");
	 				this.setWeightGradient(Vi, Ii, Wi, Connection[Ii][Ni].getActivation(Ei) * currentError[0]);
	 				
	 				shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:|>");
 				};
 			}
 			SecoIteration ApplyBiasGradient = (int Vi, int Ii)->{};
			if (isGradientSumming()) 
			{ 
				ApplyBiasGradient = (int Vi, int Ii)->
				{
					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:(Element[Vi].Dendrite[0].Bias != null && isShiftTrainable()):");
					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:(Element[Vi].Dendrite[0].Bias != null && isShiftTrainable()):");
					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>(isGradientSumming()):");
					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>=> 'Element[Vi].Dendrite[Ii].Bias':"+this.getBias(Vi, Ii)+" += 'currentError[0]':"+currentError[0]+";");
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
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:(Element[Vi].Dendrite[0].Bias != null && isShiftTrainable()):");
					    	shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>(isGradientSumming() == false):");
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>=>(isInstantGradientApply()):");
						    shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>=>=> 'Element[Vi].Dendrite[Ii].BiasGradient':"+this.getBiasGradient(Vi, Ii)+" = 'currentError[0]':"+currentError[0]+";");
						    this.setBiasGradient(Vi, Ii, currentError[0]);
					    	shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>=>=>(GradientOptimizer != null):");
					    	shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>=>=>=> double optimized = GradientOptimizer.getOptimized(Element[Vi].Dendrite[Ii].BiasGradient, Vi, Ii); ");
					    	double optimized = GradientOptimizer.getOptimized(this.getBiasGradient(Vi, Ii), Vi, Ii);
					    	shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>=>=>=> 'Element[Vi].Dendrite[Ii].Bias':"+this.getBias(Vi, Ii)+" += 'optimized':"+optimized+";");
							this.addToBias(Vi, Ii, optimized);
						};
					} 
				    else 
				    {
				    	ApplyBiasGradient = (int Vi, int Ii)->
						{
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:(Element[Vi].Dendrite[0].Bias != null && isShiftTrainable()):");
					    	shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>(isGradientSumming() == false):");
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>=>(isInstantGradientApply()):");
						    shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>=>=> 'Element[Vi].Dendrite[Ii].BiasGradient':"+this.getBiasGradient(Vi, Ii)+" = 'currentError[0]':"+currentError[0]+";");
						    //this.getBiasGradient(Vi, Ii) = currentError[0];
					    	this.setBiasGradient(Vi, Ii, currentError[0]);
						    shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>=>=>(GradientOptimizer == null):");
					    	shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>=>=>=> 'Element[Vi].Dendrite[Ii].Bias':"+this.getBias(Vi, Ii)+" += 'Element[Vi].Dendrite[Ii].BiasGradient':"+this.getBiasGradient(Vi, Ii)+";");
					        this.addToBias(Vi, Ii, this.getBiasGradient(Vi, Ii));
					        
						};
				    }
				}
				else
				{   
				    ApplyBiasGradient = (int Vi, int Ii)->	
				    {
						shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:(Element[Vi].Dendrite[0].Bias != null && isShiftTrainable()):");
						shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>=>(isInstantGradientApply()==false):");
						shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>=>=> 'Element[Vi].Dendrite[Ii].BiasGradient':"+this.getBiasGradient(Vi, Ii)+" = 'currentError[0]':"+currentError[0]+";");
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
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:(!isGradientSumming()):");
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:=>(isInstantGradientApply() && weightsAreSet && isWeightTrainable()):");
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:=>=>(GradientOptimizer != null):");
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:=>=>=> double optimized = GradientOptimizer.getOptimized(Element[Vi].Variable[Ii].WeightGradient[Wi],Ii,Wi); ");
							double optimized = GradientOptimizer.getOptimized(this.getWeightGradient(Vi, Ii, Wi),Vi,Ii,Wi);
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:=>=>=> Element[Vi].Variable[Ii].Weight[Ni] += optimized;");
							this.addToWeight(Vi, Ii, Wi, optimized);
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:=>=>|> ");
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:=>|>");
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:|>");
						};					
					} 
					else 
					{
						ApplyWeightGradient = (int Vi, int Ii, int Ni, int Ei, int Wi)->
						{
							if(this.hasWeight(Vi, Ii)==false) {return;}
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:(!isGradientSumming()):");
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:=>(isInstantGradientApply() && weightsAreSet && isWeightTrainable()):");
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:=>=>(GradientOptimizer == null):");
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:=>=>=> Element[Vi].Variable[Ii].Weight[Wi] += Element[Vi].Variable[Ii].WeightGradient[Wi];");
							this.addToWeight(Vi, Ii, Wi, this.getWeightGradient(Vi, Ii, Wi));
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:=>=>|> ");
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:=>|>");
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:|>");
						};
					}
				}
			}			
			//=========================
			// Start:
			for(int Vi=0; Vi<this.size(); Vi++) 
			{
				deriviationOffset[0] = 0;
				shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|: for(int Ii = 0; Ii < Connection.length; Ii++):");
				for (int Ii = 0; Ii < Connection.length; Ii++) 
				{
					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|: ...");
					currentError[0] = 0.0;
					CalculateInputError.on(Vi, Ii);
					// Shift/Input Gradient Calculation:
					// ---------------------------------------------------------------------------------------------------
					if (Element[Vi].hasBias(Ii) && biasIsTrainable())//  && InputIsSleeping==false // && check == true
					{
						ApplyBiasGradient.on(Vi, Ii);
					}
					// Weight gradient calculation:
					// ---------------------------------------------------------------------------------------------------
					//if (Weight != null || this.isSuperRootNode()) {// && check == true
					//What if this is firstlayer? -> firstlayer neuron doesn't get this far! -> (except it is root node (root input node)!)
			
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
								if(Element[vi].Variable[ii].Weight!=null) 
								{
									shoutLine("-rootBackward(...)|: ('Vi':"+vi+")|:('Ii':"+ii+")|:=>('Ni':"+ni+")|:('Wi':"+wi+",'Ei':"+ei+")|:=>=>=>(weightsAreSet):");
									shoutLine("-rootBackward(...)|: ('Vi':"+vi+")|:('Ii':"+ii+")|:=>('Ni':"+ni+")|:('Wi':"+wi+",'Ei':"+ei+")|:=>=>=> Connection[Ii][Ni].addToError(Ei, 'currentError[0]':"+currentError[0]+" * Element[Vi].Variable[Ii].Weight[Wi]);");
									Connection[ii][ni].addToError(ei, currentError[0] * this.getWeight(vi, ii, wi));
									shoutLine("-rootBackward(...)|: ('Vi':"+vi+")|:('Ii':"+ii+")|:=>('Ni':"+ni+")|:('Wi':"+wi+",'Ei':"+ei+")|:=>=>=>|>");
								}
								else 
								{
									shoutLine("-rootBackward(...)|: ('Vi':"+vi+")|:('Ii':"+ii+")|:=>('Ni':"+ni+")|:('Wi':"+wi+",'Ei':"+ei+")|:=>=>=>(weightsAreSet==false):");
									shoutLine("-rootBackward(...)|: ('Vi':"+vi+")|:('Ii':"+ii+")|:=>('Ni':"+ni+")|:('Wi':"+wi+",'Ei':"+ei+")|:=>=>=> Connection[Ii][Ni].addToError(Ei, 'currentError[0]':"+currentError[0]+" * 1);");
									Connection[ii][ni].addToError(ei, currentError[0] * 1);
									shoutLine("-rootBackward(...)|: ('Vi':"+vi+")|:('Ii':"+ii+")|:=>('Ni':"+ni+")|:('Wi':"+wi+",'Ei':"+ei+")|:=>=>=>|>");
								}
								// Distributing error to previous layer!
								// ================================================================================================================
								shoutLine("-rootBackward(...)|: ('Vi':"+vi+")|:('Ii':"+ii+")|:=>('Ni':"+ni+")|:('Wi':"+wi+",'Ei':"+ei+")|:=>=>|>");
							};
						}	
						shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:(Connection[Ii]!=null):");
						shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=> int Wi=0;");
						int Wi=0;
						shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=> for(int Ni = 0; Ni < Connection[Ii].length; Ni++):");
						for (int Ni = 0; Ni < Connection[Ii].length; Ni++) 
						{
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|: Connection[Ii][Ni]=Connection[Ii][Ni].Node();");
							Connection[Ii][Ni]=Connection[Ii][Ni].asNode();
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|: int limit = 'Connection[Ii][Ni].size()':"+Connection[Ii][Ni].size()+"+'Wi':"+Wi+";");
							int limit = Connection[Ii][Ni].size()+Wi;
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|: int Ei = 0;");
							int Ei = 0;
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|: while(Wi<limit):");	
				     		while(Wi<limit)
				            {     
				     			CalculateWeightGradient.on(Vi, Ii, Ni, Ei, Wi);
				     			shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:=>=> Distributing gradients now:");
				     			//TODO: implememt with if(isRootConnection(Node[] con))
				     			distributeError.on(Vi, Ii, Ni, Ei, Wi);
				     			// APPLYING WEIGHT GRADIENTS:
				     			shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:=>('Ni':"+Ni+")|:('Wi':"+Wi+",'Ei':"+Ei+")|:=>=> applying weight gradients now:");
								ApplyWeightGradient.on(Vi, Ii, Ni, Ei, Wi);		
				     			Wi++;
								Ei++;
				            }//-----------------
						 } // Input neuron loop closed!
						 shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:('Ii':"+Ii+")|:|>");
					}//Ii connection != null
					// ---------------------------------------------------------------------------------------------------
				} // Input loop closed!
			
				//GRADIENT DISTRIBUTION:
				// if this is coreheader -> nodes = getRimNodes
				if(this.isFirst()==false)
				{//First layer nodes DO NOT DISTRIBUTE GRADIENTS! (never)
					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:(this.isFirst()==false):");
					if(State.isParentRoot(this) || State.isMemoryChild(this))
					{
						shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>(State.isParentRoot(this) || State.isMemoryChild(this)):");
						shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=> Tip = NVUtility.identityCorrected.order(Tip); ");
						Tip = NVUtility.identityCorrected.order(Tip); 
						shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=> int Ti = 0;");
						int Ti = 0;
						
						int Ri = 0;
						double[] relationalDerivative = Element[Vi].getRelationalDerivative();//WIP
						
						shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=> while (Ti < Tip.length):");
						while (Ti < Tip.length) 
						{
							shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|: ...");
							if(Tip[Ti]!=null)
							{
								shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:(Tip[Ti]!=null):");
								if (State.isParentRoot(Tip[Ti].asCore()))
								{//==================================
									shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>(State.isParentRoot(Tip[Ti].Core())):");
									shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>=> Tip[Ti].addToError(Ei, Element[Vi].Output()[1] * Element[Vi].Data[2][deriviationOffset[0]]);");
									Tip[Ti].addToError(Vi, Data[0].getError(Vi) * relationalDerivative[Ri]);//ERROR DISTRIBUTION!
									shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>=> deriviationOffset[0]++;");
									Ri++;//deriviationOffset[0]++;
									shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>|>");
								}//==================================
								else if(State.isTrainableBasicChild(Tip[Ti].asCore()))
								{//==================================//ERROR HERE....
									shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>(State.isTrainableBasicChild(Tip[Ti].Core())):");
									shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>=> Tip[Ti].setError(Ei, Element[Vi].Data[2][deriviationOffset[0]]);");
									Tip[Ti].setError(Vi, relationalDerivative[Ri]);
									shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>=> deriviationOffset[0]++;");		
									Ri++;//deriviationOffset[0]++;
									shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>=> double[] targetedNodeGradient = new double[Tip[Ti].Core().inputSize()];");				
									double[] targetedNodeGradient = new double[Tip[Ti].asCore().inputSize()];
									for (int Ci = 0; Ci < Tip[Ti].asCore().inputSize() && Ri<relationalDerivative.length; Ci++) 
									{
										shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>=>('Ci':"+Ci+")|: targetedNodeGradient[Ci]=Element[Vi].Data[2][deriviationOffset[0]]; ");				
										targetedNodeGradient[Ci]=relationalDerivative[Ri]; 
										
										shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>=>('Ci':"+Ci+")|: deriviationOffset[0]++;");				
										Ri++;//deriviationOffset[0]++;
									}
									shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>=> Tip[Ti].Core().setInputGradient(targetedNodeGradient, Vi);");				
									Tip[Ti].asCore().setInputDerivative(Vi, targetedNodeGradient);
									shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>=> Tip[Ti].Executable().Backward(Hi);");							
									Tip[Ti].asExecutable().Backward(Hi);
									shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>|>");
								}//==================================
								else if(State.isMemoryChild(Tip[Ti].asCore()))
								{//==================================
									shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>(State.isMemoryChild(Tip[Ti].Core())):");
									//... add to error ...
									//find synchronization compartment -> register
									//if i am the last one to synchronize -> finish backpropagation!
									shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:=>|>");
								}//==================================
								shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>=>('Ti':"+Ti+")|:|>");
							}
							Ti++;
						}
						shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:=>|>");
					}
					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|:|>");
				}
			}//VI loop closed!
			
			if(this.isFirst()==false && (this.weightIsTrainable() || this.biasIsTrainable())) 
			{
				for(int Vi=0; Vi<this.size(); Vi++) 
				{
					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|: Element[Vi].Data[1]=null;");
					//Element[Vi].Data[1]=null;
					this.setInput(Vi, null);
					//This is important: why? : => weights changed -> input NEEDS to be recalculated!
				}
			}
			if (isInstantGradientApply() && isGradientSumming()) 
			{
				applyGradients();
			}
			if (isGradientSumming() == false) 
			{
				nullGradients();
			}
			if(this.is(BasicRoot) == false)
			{
				for(int Vi=0; Vi<this.size(); Vi++) 
				{		
					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|: CloakData[Vi][1]=0;");
					shoutLine("-rootBackward(...)|: ('Vi':"+Vi+")|: Element[Vi].setError(0);");
					//CloakData[Vi][1]=0;
					Data[0].setError(Vi, 0);
					// this.setError(Vi, 0);
				}
			}
			shoutLine("-rootBackward(...)|: END\n");
		}
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//NVNeuron implemention ends here!!!!!
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//===============================================================================================================================================
		/*
		 * tip units identifier: Are root units! What else makes them 'tip' nodes?:
		 *
		 * Either they are 'input units' (root input) Or they are memory dependent (don't have
		 * memory compartment but have trainable weights and/or biases!)... Or they are so
		 * called 'Core Header' -> meaning that they have memory -> so they
		 * automatically store derivatives with respect to connected nodes...
		 * 
		 * Do i want to have root nodes that store activations? ->YES!
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
		// needs to start->Backpropdata has been set}
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
						    	 if(current>deepest ) {deepest = current;}
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
						    		if(!Core.isFirst() || Element[Vi].hasInput()==false) 
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
										Element[Vi].setInput(null);
									}
									Element[Vi].setInputDerivative(null);
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
		@Override //NRoot //TODO implement relational derivatives as its own data field!(within element)
		public synchronized double[] findRootDerivativesOf(NVNode connectionNode, boolean inputDerivative, int Vi) 
		{// Finding derivative with respect to a specific neuron	
			shoutLine("NVertex_Root->findRootDerivativesOf(NVNode connectionNode, boolean inputDerivative, int Vi)|: ...");
			shoutLine("========================================================================");
			shoutLine("-findRootDerivativesOf(...)|: double[] resultDerivatives = null;");
			shoutLine("-findRootDerivativesOf(...)|: boolean thisIsTargNode = false;");
			double[] resultDerivatives = null;
			boolean thisIsTargNode = false;
			
			shoutLine("-findRootDerivativesOf(...)|:(Property != null):");
			if (connectionNode.asNode() == this.asNode()) 
			{
				shoutLine("-findRootDerivativesOf(...)|:=>(connectionNode.Node() == this.Node()):");
				shoutLine("-findRootDerivativesOf(...)|:=>=> thisIsTargNode=true;");
				thisIsTargNode=true;
				shoutLine("-findRootDerivativesOf(...)|:=>|>");
			}
			shoutLine("-findRootDerivativesOf(...)|:|>");
			
				
			if (thisIsTargNode)//Target node found: -> returning derivative!
			{
				shoutLine("-findRootDerivativesOf(...)|:(thisIsTargNode):");
				if (inputDerivative==false)
				{//Derivative with respect to activation. (memory nodes) 
					shoutLine("-findRootDerivativesOf(...)|:=>(inputDerivative==false):");
					shoutLine("-findRootDerivativesOf(...)|:=>=> resultDerivatives = new double[1];");
					shoutLine("-findRootDerivativesOf(...)|:=>=> resultDerivatives[0] = 1;");
					resultDerivatives    = new double[1];
					resultDerivatives[0] = 1;
					shoutLine("-findRootDerivativesOf(...)|:=>|>");
				} 
				else
				{//Derivative with respect to inputs.
					shoutLine("-findRootDerivativesOf(...)|:=>(inputDerivative!=false):");
					shoutLine("-findRootDerivativesOf(...)|:=>=> resultDerivatives = new double[Element[Vi].Data[2].length];");
					resultDerivatives = new double[Element[Vi].getInputDerivative().length];
					shoutLine("-findRootDerivativesOf(...)|:=>=> for(int Di = 0; Di < resultDerivatives.length; Di++):");
					for(int Di = 0; Di < resultDerivatives.length; Di++) 
					{
						shoutLine("-findRootDerivativesOf(...)|:=>=>('Di':"+Di+"): resultDerivatives[Di] = 'Element[Vi].Data[2][Di]'"+Element[Vi].getInputDerivative(Di)+";");
						resultDerivatives[Di] = Element[Vi].getInputDerivative(Di);
					}
					shoutLine("-findRootDerivativesOf(...)|:=>|>");
				}
				shoutLine("-findRootDerivativesOf(...)|:=> return resultDerivatives;");
				shoutLine("-findRootDerivativesOf(...)|:=> END\n");
				return resultDerivatives;
			} 
			//if(this.isInputRootNode()) {return null;}
			// why is the above not used? -> input root nodes are now not that obvious anymore -> input root define themselves
			// by havin AT LEAST ONE reference to the outside!

			if (inputDerivative==false) //Why is that? -> super root stores the relational gradient!
			{
				shoutLine("-findRootDerivativesOf(...)|:(inputDerivative==false):");
				shoutLine("-findRootDerivativesOf(...)|:=> resultDerivatives = new double[1];");
				resultDerivatives = new double[1];
				shoutLine("-findRootDerivativesOf(...)|:|>");
			} 
			else 
			{
				shoutLine("-findRootDerivativesOf(...)|:(inputDerivative==true):");
				shoutLine("-findRootDerivativesOf(...)|:=> resultDerivatives = new double[connectionNode.Core().inputSize()];");
				resultDerivatives = new double[connectionNode.asCore().inputSize()];
				shoutLine("-findRootDerivativesOf(...)|:|>");
			}
			shoutLine("-findRootDerivativesOf(...)|: for(int Ii = 0; Ii < Connection.length; Ii++): ...");
			for(int Ii = 0; Ii < Connection.length; Ii++) 
			{
				shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|: ...");
				if (Connection[Ii] != null) 
				{
					shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:(Connection[Ii] != null):");
					
					if(State.isParentRoot(Connection[Ii][0].asCore())==false && State.isRoot(Connection[Ii][0].asCore()))
					{
						shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>(State.isParentRoot(Connection[Ii][0].Core())==false && State.isRoot(Connection[Ii][0].Core())):");
						shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=> double[] inputSumDerivatives = new double[resultDerivatives.length];");
						double[] inputSumDerivatives = new double[resultDerivatives.length];
						shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=> for(int Ni = 0; Ni < Connection[Ii].length; Ni++): ...");
						for(int Ni = 0; Ni < Connection[Ii].length; Ni++) 
						{
							shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Ni':"+Ni+")|: Connection[Ii][Ni]=Connection[Ii][Ni].Node();");
							Connection[Ii][Ni]=Connection[Ii][Ni].asNode();

							if (Connection[Ii][Ni] != null) 
							{
								shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Ni':"+Ni+")|:(Connection[Ii][Ni] != null):");
								shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Ni':"+Ni+")|:=> double[] rootRelationalDerivatives = null;");
								double[] rootRelationalDerivatives = null;
								
								if(Connection[Ii][Ni]!=this && Connection[Ii][Ni]!=this.asNode()) 
								{
									shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Ni':"+Ni+")|:=>(Connection[Ii][Ni]!=this && Connection[Ii][Ni]!=this.Node()):");
									shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Ni':"+Ni+")|:=>=> rootRelationalDerivatives = Connection[Ii][Ni].Core().Root().findRootDerivativesOf(connectionNode, inputDerivative, Vi);");
									rootRelationalDerivatives = 
											Connection[Ii][Ni].asCore().asRoot().findRootDerivativesOf(connectionNode, inputDerivative, Vi);
									shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Ni':"+Ni+")|:=>|>");
								}
								
								if(rootRelationalDerivatives != null) 
								{
									shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Ni':"+Ni+")|:=>(rootRelationalDerivatives != null):");
									shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Ni':"+Ni+")|:=>=> for(int Di = 0; Di < rootRelationalDerivatives.length; Di++):");
									for(int Di = 0; Di < rootRelationalDerivatives.length; Di++) 
									{
										shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Ni':"+Ni+")|:=>=>('Di':"+Di+")|: ...");
										if(Element[Vi].Variable[Ii].Weight != null) 
										{
											shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Ni':"+Ni+")|:=>=>('Di':"+Di+")|:(Element[Vi].Variable[Ii].Weight != null):");
											shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Ni':"+Ni+")|:=>=>('Di':"+Di+")|:=> rootRelationalDerivatives[Di] *= Element[Vi].Variable[Ii].Weight[Ni];");
											rootRelationalDerivatives[Di] *= Element[Vi].Variable[Ii].Weight[Ni];
											//Why not Wi? -> 1:1 relations within root structure!
											shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Ni':"+Ni+")|:=>=>('Di':"+Di+")|:|>");
										}
										shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Ni':"+Ni+")|:=>=>('Di':"+Di+")|: inputSumDerivatives[Di] += rootRelationalDerivatives[Di];");
										inputSumDerivatives[Di] += rootRelationalDerivatives[Di];
									}
									shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Ni':"+Ni+")|:=>|>");
								}
								shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Ni':"+Ni+")|:|>");
							}
						}
						shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=> for(int Di = 0; Di < resultDerivatives.length; Di++): ...");
						for(int Di = 0; Di < resultDerivatives.length; Di++) 
						{
							shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Di':"+Di+")|:");
							if (Element[Vi].getInputDerivative() != null) 
							{
								shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Di':"+Di+")|:(Element[Vi].Data[2] != null):");
								shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Di':"+Di+")|:=> inputSumDerivatives[Di] *= Element[Vi].Data[2][Ii];");
								inputSumDerivatives[Di] *= Element[Vi].getInputDerivative(Ii);
								shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Di':"+Di+")|:|>");
							}
							//if (Element[Vi].Data[1] != null) 
							//{
							//	inputSumDerivatives[Di] *= Element[Vi].Input()[Ii];// This might not be right!
							//}
							shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>=>('Di':"+Di+")|: resultDerivatives[Di] += inputSumDerivatives[Di];");
							resultDerivatives[Di] += inputSumDerivatives[Di];
						}
						// =============
						shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:=>|>");
					}
					shoutLine("-findRootDerivativesOf(...)|: ('Ii':"+Ii+")|:|>");
				}
			}// Ii loop closed
			shoutLine("-findRootDerivativesOf(...)|: return resultDerivatives;");
			shoutLine("-findRootDerivativesOf(...)|: END\n");
			return resultDerivatives;
		}
		/*
		 * What is the use case of the above function?
		 * 
		 * The function is not in use as of yet. It allows one to get the derivative of
		 * an addressed node in a directed nodegraph. A recursive graph structure will break this function.
		 * I'm planning on using it as a
		 * tool to collect derivatives of so called "sub neurons". These sub-Neurons are
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
			shoutLine("-distributeRootRevival()|: NVNode node;");
			NVNode node;
			shoutLine("-distributeRootRevival()|: for (int Ii = 0; Ii < Connection.length; Ii++): ...");
			for (int Ii = 0; Ii < Connection.length; Ii++) 
			{
				shoutLine("-distributeRootRevival()|: ('Ii':"+Ii+")|: ...");
				if (Connection[Ii] != null) 
				{
					shoutLine("-distributeRootRevival()|: ('Ii':"+Ii+")|:(Connection[Ii] != null):");
					if(State.isParentRoot(Connection[Ii][0].asCore())==false && State.isRoot(Connection[Ii][0].asCore())==true)
					{
						shoutLine("-distributeRootRevival()|: ('Ii':"+Ii+")|:=>(State.isParentRoot(Connection[Ii][0].Core())==false && State.isRoot(Connection[Ii][0].Core())==true):");
						shoutLine("-distributeRootRevival()|: ('Ii':"+Ii+")|:=>=> setParalyzed(false);");
						setParalyzed(false);
						if(Connection[Ii]!=null) 
						{
							shoutLine("-distributeRootRevival()|: ('Ii':"+Ii+")|:=>=>(Connection[Ii]!=null):");
							shoutLine("-distributeRootRevival()|: ('Ii':"+Ii+")|:=>=>=> for(int Ni = 0; Ni < Connection[Ii].length; Ni++):");
							for(int Ni = 0; Ni < Connection[Ii].length; Ni++) 
							{
								shoutLine("-distributeRootRevival()|: ('Ii':"+Ii+")|:=>=>=>('Ni':"+Ni+")|: Connection[Ii][Ni]=Connection[Ii][Ni].Node();");
								shoutLine("-distributeRootRevival()|: ('Ii':"+Ii+")|:=>=>=>('Ni':"+Ni+")|: node = Connection[Ii][Ni].Node();");
								shoutLine("-distributeRootRevival()|: ('Ii':"+Ii+")|:=>=>=>('Ni':"+Ni+")|: node.Root().distributeRootRevival();");
								Connection[Ii][Ni]=Connection[Ii][Ni].asNode();
								node = Connection[Ii][Ni].asNode();
								node.asRoot().asParent().distributeRootRevival();
							}
							shoutLine("-distributeRootRevival()|: ('Ii':"+Ii+")|:=>=>|>");
						}
						shoutLine("-distributeRootRevival()|: ('Ii':"+Ii+")|:=>|>");
					}
					shoutLine("-distributeRootRevival()|: ('Ii':"+Ii+")|:|>");
				}
			}
			shoutLine("-distributeRootRevival()|: END\n");
		}
		// NOTE: This function will only produce new output if recalculation permission
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
			if (Element[Vi].getInput()!=null && isParalyzed()==false) //This is not entered wehen vi==1
			{
				shoutLine("-findRootActivation(...)|: (Element[Vi].Input()!=null && isParalyzed()==false):");
				shoutLine("-findRootActivation(...)|: => boolean[] Signal = {true};");
				boolean[] Signal;
				if(this.is(NVertex.RootInput)) 
				{
					shoutLine("-findRootActivation(...)|: =>(this.is(NVertex.RootInput)):");
					shoutLine("-findRootActivation(...)|: =>=> Signal = this.inputSignalIf(true);");
					Signal = this.inputSignalIf(true);
				}
				else
				{
					if(Element[Vi].hasInput()==false)// This might not be right....
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
				shoutLine("-findRootActivation(...)|: => Element[Vi].Data = activationOf(weightedConvectionOf(publicized(Signal),Vi),Vi);");
				Element = activationOf(weightedConvectionOf(publicized(Signal),Vi),Vi).getDataAsElementArray();//publicized is not needed! 
				//shoutLine("-findRootActivation(...)|: => setParalyzed(true);");
				//setParalyzed(true);
			}
			shoutLine("-findRootActivation(...)|: => return Element[Vi].Output()[0];");
			return Element[Vi].getActivation();
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
			if(Parent==null || Parent==this) {return true;}
			return false;
		}
		@Override //NVRoot
		public NVParent asParent() 
		{
			if(Parent==null || Parent==this) 
			{
				return this;
			}
			return null;
		}
		@Override //NVRoot
		public boolean isChild() 
		{
			if(Parent==null || Parent==this)
			{
				return false;
			}
			return true;
		}
		@Override //NVRoot
		public NVChild asChild() 
		{
			if(Parent==null || Parent==this)
			{
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
		return Signal;
	}
		
}//Class end.
//=========================================================================================================
