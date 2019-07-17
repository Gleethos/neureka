
package neureka.main.core;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Random;

import neureka.main.core.modul.calc.FunctionConstructor;
import neureka.main.core.modul.calc.Function;
import neureka.main.core.base.NVData;
import neureka.main.core.base.NVNeuron;
import neureka.main.core.base.NVInput;
import neureka.main.core.imp.NVertex_Root;
import neureka.main.core.modul.mem.NVMemory;
import neureka.main.core.modul.opti.NVOptimizer;
import neureka.main.exec.cpu.NVExecutable;
import neureka.ngui.NVControlFrame;

public class NVertex extends NVData implements NVNode
{	
	private ArrayList<Object> Modules = new ArrayList<Object>();
	//-------------------------------------------------------------
	int Flags=16+32+64;//Default
	private final static int 
	isFirst_MASK = 1,
	isHidden_MASK = 2,
	isLast_MASK = 4,
	isParalyzed_MASK = 8,
	weightIsTrainable_MASK = 16,
	biasIsTrainable_MASK = 32,
	applyGradientInstantly_MASK = 64,
	sumGradient_MASK = 128;
	//---------------------------------------------
	protected interface QuinIteration{public void on(int prim, int seco, int tert, int quart, int quint);}
	protected interface QuarIteration{public void on(int prim, int seco, int tert, int quart);};
	protected interface TertIteration {public void on(int prim, int seco, int tert);};
	protected interface SecoIteration {public void on(int prim, int seco);};
	protected interface PrimIteration {public void on(int prim);}

	protected long ID = new Random().nextLong();

	protected NVNode[][] Connection; // [Value size][connections]
	protected byte[]     InputState;
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public static Typer Root = 				 (NVNode node)->State.isRoot(node);
	public static Typer BasicRoot = 		 (NVNode node)->State.isBasicChild(node);
	public static Typer RootMemory = 		 (NVNode node)->State.isMemoryChild(node);
	public static Typer RootInput = 		 (NVNode node)->State.isInputRoot(node);
	public static Typer TrainableBasicRoot =(NVNode node)->State.isTrainableBasicChild(node);
	public static Typer TrainableNeuron = 	 (NVNode node)->State.isTrainableNeuron(node);
	public static Typer RootTip  = 			 (NVNode node)->State.isRootTip(node);
	public static Typer MotherRoot = 		 (NVNode node)->State.isParentRoot(node);
	public static Typer Child = 		     (NVNode node)->State.isChild(node);
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	//MODUL I / O :
	//=========================
	public ArrayList<Object> getModules()
	{
		return Modules;
	}
	public void setModules(ArrayList<Object> properties)
	{
		Modules = properties;
	}
	public void addModule(Object newModule)
	{
		if(Modules != null)
		{
			Object oldCompartment = findModule(newModule.getClass());
			if(oldCompartment != null) {Modules.remove(oldCompartment);Modules.trimToSize();}
			Modules.add(newModule);
		}
	}
	public Object findModule(Class moduleClass)
	{
		if(Modules != null)
		{
			for(int Pi=0; Pi<Modules.size(); Pi++)
			{
				if(moduleClass.isInstance(Modules.get(Pi)))
				{
					return Modules.get(Pi);
				}
			}
		}
		return null;
	}
	public void removeModule(Class moduleClass)
	{
		Object oldCompartment = findModule(moduleClass);
		if(oldCompartment!=null) {Modules.remove(oldCompartment);Modules.trimToSize();}
	}
	public boolean hasModule(Class moduleClass)
	{
		if(findModule(moduleClass) != null) {return true;}
		return false;
	}
	//====================================================================================================
	//BOOLEAN I / O properties:
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public boolean isParalyzed(){if((Flags&isParalyzed_MASK)==isParalyzed_MASK) {return true;}return false;}
	public void    setParalyzed(boolean paralyzed) {if(isParalyzed()!=paralyzed) {if(paralyzed) {Flags+=isParalyzed_MASK;}else{Flags-=isParalyzed_MASK;}}}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public boolean isFirst(){if((Flags&isFirst_MASK)==isFirst_MASK) {return true;}return false;}
	public void    setFirst(boolean isFirst) {if(isFirst()!=isFirst) {if(isFirst) {Flags+=isFirst_MASK;}else{Flags-=isFirst_MASK;}}}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public boolean isHidden(){if((Flags&isHidden_MASK)==isHidden_MASK) {return true;}return false;}
	public void     setHidden(boolean isHidden) {if(isHidden()!=isHidden) {if(isHidden) {Flags+=isHidden_MASK;}else{Flags-=isHidden_MASK;}}}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public boolean isLast(){if((Flags&isLast_MASK)==isLast_MASK) {return true;}else{return false;}}
	public void    setLast(boolean isLast)  {if(isLast()!=isLast) {if(isLast) {Flags+=isLast_MASK;}else{Flags-=isLast_MASK;}}}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//TRAINING BEHAIVIOR properties:
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public boolean weightIsTrainable(){if((Flags&weightIsTrainable_MASK)==weightIsTrainable_MASK) {return true;}return false;}
	public void    setWeightTrainable(boolean weightTrainable) {if(weightIsTrainable()!=weightTrainable) {if(weightTrainable) {Flags+=weightIsTrainable_MASK;}else{Flags-=weightIsTrainable_MASK;}}}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public boolean biasIsTrainable(){if((Flags&biasIsTrainable_MASK)==biasIsTrainable_MASK) {return true;}return false;}
	public void    setBiasTrainable(boolean shiftTrainable) {if(biasIsTrainable()!=shiftTrainable) {if(shiftTrainable) {Flags+=biasIsTrainable_MASK;}else{Flags-=biasIsTrainable_MASK;}}}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public boolean isInstantGradientApply() {if((Flags&applyGradientInstantly_MASK)==applyGradientInstantly_MASK) {return true;}return false;}
	public void    setInstantGradientApply(boolean instantGradientApply) {if(isInstantGradientApply()!=instantGradientApply){if(instantGradientApply) {Flags+=applyGradientInstantly_MASK;}else{Flags-=applyGradientInstantly_MASK;}}}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public boolean isGradientSumming() {if((Flags&sumGradient_MASK)==sumGradient_MASK) {return true;}return false;}
	public void    setGradientSumming(boolean gradientSumming) {if(isGradientSumming()!=gradientSumming) {if(gradientSumming) {Flags+=sumGradient_MASK;}else{Flags-=sumGradient_MASK;}}}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//================================================================================================
	@Override
	public boolean is(Typer type) {return type.check(this);}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override//NVNode
	public int size() {return Neuron.length;}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NVNode
	public void apply(NAction Actor) {Actor.act(this);}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NVNode
	public NVNode asNode(){return this;}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NVNode
	public NVRoot asRoot() {if(this instanceof NVRoot) {return (NVRoot)this;}return null;}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NCore_Neuron
	public neureka.main.core.NVNeuron asNeuron() {if(this instanceof neureka.main.core.NVNeuron) {return (neureka.main.core.NVNeuron)this;}return null;}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NVNode
	public NVExecutable asExecutable() 
	{
		if(this instanceof NVExecutable) 
		{
			return (NVExecutable)this;
		}
		return null;
	}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NVNode
	public NVNode[] find(NCondition condition) 
	{
		NVNode[] checkedNodes = new NVNode[0];
		boolean check = condition.check(this);
		if (check) 
		{
			checkedNodes = new NVNode[1];checkedNodes[0] = this;
		}
		//if (this.isRootInputNode()) {if(checkedNodes[0]==null) {checkedNodes=null;}return checkedNodes;}
		if (Connection != null) 
		{
			for (int Ii = 0; Ii < inputSize(); Ii++)
			{
				if (Connection[Ii] != null)
				{
					if(State.isParentRoot(Connection[Ii][0].asCore())==false && State.isRoot(Connection[Ii][0].asCore()))
					{
						for (int Ni = 0; Ni < Connection[Ii].length; Ni++) 
						{
							if(Connection[Ii][Ni]!=null) // Maybe not needed!
							{
								if(Connection[Ii][Ni]!=this) 
								{
									NVNode[] justChecked = Connection[Ii][Ni].asCore().find(condition);
									if (justChecked != null) 
									{
										NVNode[] newlyChecked = new NVNode[checkedNodes.length + justChecked.length];
										for (int i = 0; i < checkedNodes.length; i++)
										{
											newlyChecked[i] = checkedNodes[i];
										}
										for (int i = 0; i < justChecked.length; i++)
										{
											newlyChecked[i + checkedNodes.length] = justChecked[i];
										}
										checkedNodes = newlyChecked;
									}
								}
							}
						}
					}
				}
			}
		}
		return checkedNodes;
	}
	// CONSTRUCTORS:
	// ============================================================================================================================================================================================
	public NVertex(int size, boolean isFirst, boolean isLast, int v_id, int f_id)
	{
		super(null);
		construct(size, isFirst, isLast, true,  v_id, f_id, 1);
	}
	public NVertex(int size, boolean isFirst, boolean isLast, int v_id, int f_id, int delay) 
	{
		super(null);
		construct(size, isFirst, isLast, true, v_id, f_id, delay);
	}
	public NVertex(NVNode image, boolean isFirst, boolean isLast, int v_id, int f_id, int delay) 
	{
		super(null);
		int size = 1;
		if(image!=null) 
		{
			size=image.size();
		}
		construct(size, isFirst, isLast, true, v_id, f_id, delay);
	}
	public NVertex(NVertex toBeCopied)
	{
		super(null);
		copy(toBeCopied);
	}
	//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	private void construct(int size, boolean isFirst, boolean isLast, boolean derivable, long v_id, int f_id, int delay) 
	{
		shoutLine("new NVertex(int size="+size+", boolean isFirst="+isFirst+", boolean isLast="+isLast+", int v_id="+v_id+", int f_id="+f_id+", int delay="+delay+")|: ...created!");
		if(size<=0) 
		{
			size=1;
		}
		ID = v_id;
		shoutLine("Neuron creation!");
		setParalyzed(false);
		Neuron = new NVNeuron[size];
		
		for(int Vi=0; Vi<size; Vi++) 
		{
			Neuron[Vi] = new NVNeuron(1);
			Neuron[Vi].Input = new NVInput[1];
			Neuron[Vi].Input[0] = new NVInput(0, derivable);
		}
		Connection = new NVNode[1][];
		InputState = new byte[1];
		setFirst(isFirst);
		setLast(isLast);
		if (isFirst == false && isLast == false)
		{
			setHidden(true);
		}
		else
		{
			setHidden(false);
		}
		Function newFunction;
		switch(f_id) 
		{   
		    case 0:  newFunction = new FunctionConstructor().newBuild("prod(relu(Ij))"); break;
			case 1:  newFunction = new FunctionConstructor().newBuild("prod(sig(Ij))"); break;
			case 2:  newFunction = new FunctionConstructor().newBuild("prod(tanh(Ij))"); break;
			case 3:  newFunction = new FunctionConstructor().newBuild("prod(quadr(Ij))"); break;
			case 4:  newFunction = new FunctionConstructor().newBuild("prod(lig(Ij))"); break;
			case 5:  newFunction = new FunctionConstructor().newBuild("prod(lin(Ij))"); break;
			case 6:  newFunction = new FunctionConstructor().newBuild("prod(gaus(Ij))"); break;
			default: newFunction = new FunctionConstructor().newBuild("prod(tanh(Ij))");break;
		}
		addModule(newFunction);
		if(delay>0) {addModule(new NVMemory(delay));}
		setDerivable(derivable);
		setConvergence(false);
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public String toString()
	{
		String base = "||   ";

		String S = "\n";
		S+= "[]|[=]|[=]| NODE["+ID+"]:\n";
		S+= base+"\n";
		S+= base+"ID:"+ ID + ", \n";
		S+= base+"Modules: \n";

		for(int Mi=0; Mi<this.Modules.size(); Mi++){
			String info = "";
			if(Modules.get(Mi) instanceof Function){
				if(this.hasModule(Function.class)) {info = ((Function) findModule(Function.class)).toString();}
			}
			S+= base+Modules.get(Mi).getClass()+" "+info+"\n";
		}
		S+=  base+"\n"
			+base+ "isFirst(): " + 			       isFirst()+ ", \n"
			+base+ "isHidden(): " + 			   isHidden() + ", \n"
			+base+ "isLast(): " + 				   isLast() + ", \n"
			+base+ "isParalyzed(): "+ 			   isParalyzed() + ", \n"
			+base+ "isWeightTrainable(): " +      weightIsTrainable() + ", \n"
			+base+ "isShiftTrainable(): " + 	   biasIsTrainable()+ ", \n"
			+base+ "isInstantGradientApply(): " + isInstantGradientApply() + ", \n"
			+base+ "isGradientSumming(): " + 	   isGradientSumming() + "\n"
			+base+ "isOfRootType: " + 		       State.isRoot(this) + ", \n"
			+base+ "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
		S+= super.toString()+"";
		S+= base+"\n";
		S+="[]|[=]|[=]|[=]|>\n";
		return S;
	}
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public synchronized void removeInput(int targIi) 
	{
		if (Neuron[0].Input.length >= targIi)
		{
			return;//Target index does not exist.
		}
		for(int Vi = 0; Vi< Neuron.length; Vi++)
		{
			Neuron[Vi].removeInputAt(targIi);
			//Remove target index from all elements
		}
		NVUtility<NVNode[]> utility = new NVUtility<NVNode[]>();
		Connection = utility.updateArray(Connection, targIi, true, NVNode[].class);
		InputState = utility.updateArray(InputState, targIi, true);
		//---------------------------------------
		NVOptimizer GradientOptimizer = (NVOptimizer)findModule(NVOptimizer.class);
		if (GradientOptimizer != null) 
		{
			removeModule(NVOptimizer.class);
			addModule(new NVOptimizer(Neuron.length, Connection, GradientOptimizer.getOptimizationID()));
		}
		//---------------------------------------------------------------------------------------
		NVMemory Memory = (NVMemory)findModule(NVMemory.class);
		if(Memory!=null)
		{
			Memory.resetMemory();
		}
		//---------------------------------------
	}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public synchronized void addInput(int targIi) {addInput(targIi, 1);}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public synchronized void addInput(int targIi, int Fid) 
	{
		if (Neuron[0].Input.length < targIi)
		{
			return;
		}
		for(int Di = 0; Di< Neuron.length; Di++)
		{
			Neuron[Di].addInputAt(targIi);
		}
		NVUtility<NVNode[]> utility = new NVUtility<NVNode[]>();
		Connection = utility.updateArray(Connection, targIi, false, NVNode[].class);
		InputState = utility.updateArray(InputState, targIi, false);
		for(int Ii=0; Ii<Connection.length; Ii++)
		{
			if(Connection[Ii]!=null) {
				for(int Ni=0; Ni<Connection[Ii].length; Ni++) {
					Connection[Ii][Ni].hasBias();
				}
			}
		}
		NVOptimizer GradientOptimizer = (NVOptimizer)findModule(NVOptimizer.class);
		if (GradientOptimizer != null) 
		{
			removeModule(NVOptimizer.class);
			addModule(new NVOptimizer(Neuron.length, Connection, GradientOptimizer.getOptimizationID()));
		}
		NVMemory Memory = (NVMemory)findModule(NVMemory.class);
		if(Memory!=null) {Memory.resetMemory();}
	}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public void swapConnection(NVNode former, NVNode replacement) 
	{
		for(int Di = 0; Di< Neuron.length; Di++)
		{
			former = former.asNode(); replacement = replacement.asNode();
			if(Neuron[Di].Input ==null) {return;}
			for(int Ii=0; Ii<inputSize(); Ii++) 
			{
		    	if(Connection[Ii]!=null) 
		    	{
		    		for(int Ni=0; Ni<Connection[Ii].length;Ni++) 
		    		{
		    			Connection[Ii][Ni]=Connection[Ii][Ni].asNode();
		    			if(Connection[Ii][Ni]==former) 
		    			{
		    				Connection[Ii][Ni]=replacement;
		    			}
		    		}
		    	}
			}
		}
	}
	// DISCONNECTING NODE
	// ============================================================================================================================================================================================
	public synchronized void disconnect(NVNode ConnectionNode) 
	{
		ConnectionNode = ConnectionNode.asNode();
		for (int Ii = 0; Ii < Neuron[0].Input.length; Ii++)
		{
			disconnect(ConnectionNode, Ii);
		}
	}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public synchronized void disconnect(NVNode[] array, int targIi) 
	{
		for(int i=0; i<array.length; i++) {disconnect(array[0],targIi);}
	}
	public synchronized void disconnect(NVNode ConnectionNode, int targIi) 
	{
		ConnectionNode = ConnectionNode.asNode();
		shoutLine("-disconnect(...)|: AIC[targIi:(" + targIi + ")]!");
		if (Connection == null) {return;}
		if (inputSize() <= targIi) {return;}
		if (Connection[targIi] == null) 
		{
			shoutLine("-disconnect(...)|: AIC[targIi:(" + targIi + ")] is null, empty input Data[Di].Value !");
			return;
		}
		int i = - 1;
		for(int Ni=0; Ni<Connection[targIi].length; Ni++) 
		{
			Connection[targIi][Ni]=Connection[targIi][Ni].asNode();
		    if(Connection[targIi][Ni].asCore()==ConnectionNode.asCore()) {i=Ni;}
		}
		if(i==-1) {return;}
		shoutLine("-disconnect(...)|: removing Data[Di].Value now! |: ->AIC.length:(" + inputSize()
				+ ") |: calling uptdateInputData[Di].Value(Ni:(" + i + "), remove:(false)); ");
		
		NVUtility<NVNode> utility1 = new NVUtility<NVNode>();
		Connection[targIi] = utility1.updateArray(Connection[targIi], i, true, NVNode.class);
		
		for(int Di = 0; Di< Neuron.length; Di++)
		{
			NVUtility<double[]> Utility2 = new NVUtility<double[]>();
			for(int Wi=0; Wi<ConnectionNode.size(); Wi++) 
			{
			    Neuron[Di].Input[targIi].Weight = Utility2.updateArray(Neuron[Di].Input[targIi].Weight, i, true);
			}
		}
		if (Connection[targIi] != null) 
		{shoutLine("-disconnect(...)|: new AIC -> AIC-I[" + targIi + "].length = " + Connection[targIi].length);}
		NVOptimizer oldGradientOptimizer = (NVOptimizer) findModule(NVOptimizer.class);
		if (oldGradientOptimizer != null) 
		{
			removeModule(NVOptimizer.class);
			addModule(new NVOptimizer(Neuron.length, Connection, oldGradientOptimizer.getOptimizationID()));
		}
		NVMemory Memory = (NVMemory)findModule(NVMemory.class);
		if(Memory!=null) {Memory.resetMemory();}
		return;
	}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	// CONNECTING TO NODE:
	// ============================================================================================================================================================================================
	//NVNode
	public synchronized void connect(NVNode ConnectionNode) 
	{
		ConnectionNode = ConnectionNode.asNode();
		for(int Ii = 0; Ii < Neuron[0].Input.length; Ii++)
		{
			connect(ConnectionNode, Ii);
		}
	}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//NVNode
	public synchronized void connect(NVNode[] array, int targIi) 
	{
		for(int i=0; i<array.length; i++) {connect(array[0],targIi);}
	}
	//NVNode
	public synchronized void connect(NVNode node, int targIi) 
	{
		shoutLine("NVertex->connect("+node.getClass()+" node, int " +targIi +")|: Connecting to 'Value-Index':" + targIi + "  !");
		shoutLine("========================================================================");
		node = node.asNode();
		shoutLine("-connect(..., int 'targIi':"+targIi+")|: node = node.Node();");
		if(node == null) 
		{
			shoutLine("-connect(..., int 'targIi':"+targIi+")|: (node == null):");
			shoutLine("-connect(..., int 'targIi':"+targIi+")|: => return;");
			shoutLine("-connect(..., int 'targIi':"+targIi+")|: => END\n;");
			return;
		}
		shoutLine("-connect(..., int 'targIi':"+targIi+")|: for(int Vi=0; Vi<Neuron.length; Vi++): ...");
		if (Neuron[0].Input == null)
		{
			shoutLine("-connect(..., int 'targIi':"+targIi+")|: (Neuron[0].Dendrite == null):");
			if(targIi==0) 
			{ 
				shoutLine("-connect(..., int 'targIi':"+targIi+")|: =>(targIi==0):");
				shoutLine("-connect(..., int 'targIi':"+targIi+")|: =>=> Neuron[Vi].Value = new NVElementWeightContext[1];");
				shoutLine("-connect(..., int 'targIi':"+targIi+")|: =>=> Connection = new NVNode[1][];") ;
				Neuron[0].Input = new NVInput[1];
				Connection = new NVNode[1][];
			}
			else
			{
				shoutLine("-connect(..., int 'targIi':"+targIi+")|: =>(targIi != 0):");
				shoutLine("-connect(..., int 'targIi':"+targIi+")|: =>=> return;");
				shoutLine("-connect(..., int 'targIi':"+targIi+")|: =>=>End\n") ;
				return;
			}	
		}
		else
		{
			shoutLine("-connect(..., int 'targIi':"+targIi+")|: (Neuron[Vi].Dendrite != null):");
		}
		if (targIi >= Neuron[0].Input.length)
		{
			shoutLine("-connect(..., int 'targIi':"+targIi+")|: (targIi >= Neuron[Vi].Dendrite.length):");
			shoutLine("-connect(..., int 'targIi':"+targIi+")|: => return;") ;
			shoutLine("-connect(..., int 'targIi':"+targIi+")|: End\n") ;
			return;
		}
		if(State.isRoot(this)) 
		{
			if(Connection[targIi]!=null) 
			{
				NVertex ConnCore = Connection[targIi][0].asCore();
			   
				if((State.isParentRoot(ConnCore)==true     || State.isRoot(ConnCore)==false)
				&& (State.isParentRoot(node.asCore())==false && State.isRoot(node.asCore())==true)) {return;}
			   
				if(((State.isParentRoot(ConnCore) == false && State.isRoot(ConnCore)==true)
			    && (State.isParentRoot(node.asCore())==true || State.isRoot(node.asCore())==false))){return;}
		   	   		
			}
		}
		if (Neuron[0].Input[targIi] == null)
		{
			shoutLine("-connect(..., int 'targIi':"+targIi+")|: AIC[targIi:(" + targIi + ")] is null, currently empty input Data[Di].Value !");
		}
		int i = 0;
		if(Connection[targIi]!=null) 
		{
			i=Connection[targIi].length;
		}
		shoutLine("-connect(..., int 'targIi':"+targIi+")|: Connecting now! | calling uptdateInputData[Di].Value(Ni:(" + i + "), remove:(false)); ");
		NVUtility<NVNode> utility = new NVUtility<NVNode>();
		if(Connection[targIi]==null) 
		{
			Connection[targIi] = new NVNode[0];
		}
		Connection[targIi] = utility.updateArray(Connection[targIi], i, false, NVNode.class);

		if (Connection[targIi][i] == null) 
		{
			shoutLine("-connect(..., int 'targIi':"+targIi+")|: (Connection[targIi][i] == null): ERROR!!!");
		}
		Connection[targIi][i] = node;
		if(Neuron[0].Input !=null && Neuron[0].Input[targIi].Weight!=null)
		{
			int Wi=0;
			for(int Ni=0; Ni<Connection[targIi].length; Ni++) 
			{
				Wi+=Connection[targIi][Ni].size();
			}
			for(int wi=0; wi<Wi; wi++) 
			{
				NVUtility<double[]> utility2 = new NVUtility<double[]>();
				Neuron[0].Input[targIi].Weight =  utility2.updateArray(Neuron[0].Input[targIi].Weight, i, false);
			}
		}
		if(this.is(NVertex.Root) && node.asCore().is(NVertex.Child) && Connection[targIi].length==1) 
		{
			Neuron[0].Input[targIi].Weight = null;
		}
		shoutLine("-connect(..., int 'targIi':"+targIi+")|: 'Connection[targIi].length':" + Connection[targIi].length);
		NVOptimizer GradientOptimizer = (NVOptimizer)findModule(NVOptimizer.class);
		if (GradientOptimizer != null)
		{
			shoutLine("-connect(..., int 'targIi':"+targIi+")|: (GradientOptimizer != null):");
			removeModule(NVOptimizer.class);
			addModule(new NVOptimizer(Neuron.length, Connection, GradientOptimizer.getOptimizationID()));
		}
		NVMemory Memory = (NVMemory)findModule(NVMemory.class);
		if(Memory!=null)
		{
			Memory.resetMemory();
		}
		shoutLine("-connect(..., int 'targIi':"+targIi+")|: Checking root depth:");
		if(this.is(NVertex.Root)) 
		{
			if(asRoot().rootStructureDepth(0)>100) 
			{
				shoutLine("-connect(...)|: (Root().rootDepth(0)>100) => disconnect(node, targIi);");
				disconnect(node, targIi);
			}
		}
		shoutLine("-connect(..., int 'targIi':"+targIi+")|: End\n") ;
		return;
	}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public synchronized void randomizeBiasAndWeight()
	{
		shoutLine("NVertex.randomizeShiftAndWeightMatrix|: ...!");
		for(int Vi = 0; Vi< Neuron.length; Vi++)
		{
			if (Neuron[Vi].Input == null)
			{
				shoutLine("-randomizeShiftAndWeightMatrix|: Weight randomization failed! No weight to randomize.");
			}
			else
			{
				for (int Ii = 0; Ii < Neuron[Vi].Input.length; Ii++)
				{
					if (Neuron[Vi].Input[Ii].Weight != null)
					{
						for (int Wi = 0; Wi < Neuron[Vi].Input[Ii].Weight.length; Wi++)
						{
							Neuron[Vi].Input[Ii].Weight[Wi] = NVUtility.getDoubleOf(Wi+Ii+Vi*Ii*Vi);
						}
						Neuron[Vi].Input[Ii].Bias = NVUtility.getDoubleOf(278+Ii+Vi*Ii*Vi);
					}
				}
			}
			setError(Vi,0);
			//NVCloak cloak = (NVCloak)find(NVCloak.class);
			//if(cloak!=null)
			//{
			//	cloak.setError(Vi, 0.0);
			//}
		}
	}
	// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	private void instantiateWeightMatrix()
	{
		for(int Di = 0; Di< Neuron.length; Di++)
		{
			for (int Ii = 0; Ii < Neuron[Di].Input.length; Ii++)
			{
				if (Neuron[Di].Input[Ii] != null)
				{
					int size = 0;
					for(int Ni=0; Ni<inputSize(); Ni++) {size += Connection[Ii][Ni].size();}
					Neuron[Di].Input[Ii].Weight = new double[size];
				}
			}
		}
	}
	// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public void biasAll()
	{
		for(int Di = 0; Di< Neuron.length; Di++)
		{
			for (int Ii = 0; Ii < Neuron[Di].Input.length; Ii++)
			{
				Neuron[Di].Input[Ii].Bias = 0.0;
			}
		}
	}
	// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	protected void instantiateWeightGradient()
	{
		for(int Di = 0; Di< Neuron.length; Di++)
		{
			for (int Ii = 0; Ii < Neuron[Di].Input.length; Ii++)
			{
				if (Neuron[Di].Input[Ii] != null)
				{
					if(Neuron[Di].Input[Ii].Weight!=null)
					{
						Neuron[Di].Input[Ii].WeightGradient = new double[Neuron[Di].Input[Ii].Weight.length];
					}
				}
			}
		}
	}
	// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	protected void instantiateBiasGradient()
	{
		for(int Di = 0; Di< Neuron.length; Di++)
		{
			for (int Ii = 0; Ii < Neuron[Di].Input.length; Ii++)
			{
				Neuron[Di].Input[Ii].BiasGradient = 0.0;
			}
		}
	}
	// -------------------------------------------------------------------------------------------------
	public synchronized void applyGradients()
	{
		if(weightIsTrainable()==false && biasIsTrainable()==false)
		{
			return;
		}
		if (weightIsTrainable())
		{
			instantiateWeightGradient();
		}
		if (biasIsTrainable())
		{
			instantiateBiasGradient();
		}
		for(int Vi = 0; Vi< Neuron.length; Vi++)
		{
			if(!isFirst())
			{
				Neuron[Vi].setInput(null);
			}
			NVOptimizer GradientOptimizer = (NVOptimizer)findModule(NVOptimizer.class);
			for (int Ii = 0; Ii < Neuron[Vi].Input.length; Ii++)
			{
				if (Neuron[Vi].Input[Ii].Weight != null && weightIsTrainable())
				{
					int Wi=0;
					for (int Ni = 0; Ni < Connection[Ii].length; Ni++)
					{
						if (GradientOptimizer != null)
						{
							int limit = Connection[Ii][Ni].size()+Wi;
	        		   		while(Wi<limit)
	        		   		{
	        		   			double optimized = GradientOptimizer.getOptimized(Neuron[Vi].Input[Ii].Weight[Wi], Ii, Wi);
	        		   			shoutLine("-applyGradients|: Applying optimized WG[i" + Ii + "][n" + Ni + "]:("
								+ Neuron[Vi].Input[Ii].WeightGradient[Wi] + "=>" + optimized + ") to W[i" + Ii + "][n" + Ni + "]:("
								+ Neuron[Vi].Input[Ii].Weight[Wi] + ");");
	        		   			//======================================
	        		   			Neuron[Vi].Input[Ii].Weight[Wi] += optimized;
	        		   			//======================================
	        		   			Wi++;
			     			}
						}
						else
						{
							int limit = Connection[Ii][Ni].size()+Wi;
	        		   		while(Wi<limit)
	        		   		{
	        		   			shoutLine("-applyGradients|: Applying WG[i" + Ii + "][" + Wi + "]:(" + Neuron[Vi].Input[Ii].WeightGradient[Wi]
	        		   				+ ") to W[i" + Ii + "][n" + Wi + "]:(" + Neuron[Vi].Input[Ii].Weight[Wi] + ");");
	        		   			//======================================
	        		   			Neuron[Vi].Input[Ii].Weight[Wi] += Neuron[Vi].Input[Ii].WeightGradient[Wi];
	        		   			//======================================
	        		   			Wi++;
			     			}
						}
					}
				}
				// SHIFT GRADIENTS:
				if (Neuron[Vi].Input[Ii].Bias != null && biasIsTrainable())
				{
					if (GradientOptimizer != null)
					{//======================================
						Neuron[Vi].Input[Ii].Bias += GradientOptimizer.getOptimized(Neuron[Vi].Input[Ii].BiasGradient, Vi, Ii);
					}//======================================
					else
					{//======================================
						Neuron[Vi].Input[Ii].Bias += Neuron[Vi].Input[Ii].BiasGradient;
					}//======================================
				}
			}
		}
		nullGradients();
	}

	public void nullGradients(){for(int Di = 0; Di< Neuron.length; Di++) {
		Neuron[Di].nullGradients();}}

	// ============================================================================================================================================================================================
	public void updateInputState(boolean activity, int Ii)
	{
		shoutLine("NVertex->updateInputState(boolean activity, int Ii)|: ...");
		shoutLine("========================================================================");
		shoutLine("-updateInputState(...)|: ...");
		if(InputState[Ii]<0)
		{
			shoutLine("-updateInputState(...)|: (InputState[Ii]<0):");
			if(activity) 
			{
				shoutLine("-updateInputState(...)|: =>(activity):");
				shoutLine("-updateInputState(...)|: =>=> InputState[Ii]=0;");
				InputState[Ii]=0;
				shoutLine("-updateInputState(...)|: =>|>");
			}
			else
			{
				shoutLine("-updateInputState(...)|: =>(!activity):");
				if(InputState[Ii]>-128) 
				{
					shoutLine("-updateInputState(...)|: =>=>(InputState[Ii]>-128):");
					shoutLine("-updateInputState(...)|: =>=>=> Random dice = new Random();");
					Random dice = new Random(); 
					if((dice.nextInt()%256-128)<InputState[Ii]) 
					{
						shoutLine("-updateInputState(...)|: =>=>=>((dice.nextInt()%256-128)<InputState[Ii]):");
						shoutLine("-updateInputState(...)|: =>=>=>=> InputState[Ii]--;");
						InputState[Ii]--;
						shoutLine("-updateInputState(...)|: =>=>=>|>");
					}
					shoutLine("-updateInputState(...)|: =>=>|>");
				}
				shoutLine("-updateInputState(...)|: =>|>");
			}
			shoutLine("-updateInputState(...)|: |>");
		}
		else
		{
			shoutLine("-updateInputState(...)|: (InputState[Ii]>=0):");
			if(!activity) 
			{	
				shoutLine("-updateInputState(...)|: =>(!activity):");
				shoutLine("-updateInputState(...)|: =>=> InputState[Ii]=-1;");
				InputState[Ii]=-1;
				shoutLine("-updateInputState(...)|: =>|>");
			}
			else
			{
				shoutLine("-updateInputState(...)|: =>(activity):");
				if(InputState[Ii]<127) 
				{
					shoutLine("-updateInputState(...)|: =>=>(InputState[Ii]>-128):");
					shoutLine("-updateInputState(...)|: =>=>=> Random dice = new Random();");
					Random dice = new Random(); 
					if((dice.nextInt()%256-128)>InputState[Ii]) 
					{
						shoutLine("-updateInputState(...)|: =>=>=>((dice.nextInt()%256-128)>InputState[Ii]):");
						shoutLine("-updateInputState(...)|: =>=>=>=> InputState[Ii]++;");
						InputState[Ii]++;
						shoutLine("-updateInputState(...)|: =>=>=>|>");
					}
					shoutLine("-updateInputState(...)|: =>=>|>");
				}
				shoutLine("-updateInputState(...)|: =>|>");
			}
			shoutLine("-updateInputState(...)|: |>");
		}
		shoutLine("-updateInputState(...)|: END\n");
	}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public synchronized long getID() {return ID;}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public synchronized Function getFunction() {return (Function)findModule(Function.class);}
	public synchronized void 	    setFunction(Function function) {addModule(function);}
	public synchronized NVNode[][] getConnection() {return Connection;}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public synchronized void       setShiftTrainability(boolean trainable) {setBiasTrainable(trainable);}
	public synchronized void       setWeightTrainability(boolean trainable) {setWeightTrainable(trainable);}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	public boolean ActivitySignalAt(BigInteger Hi) 
	{
		//This CorefindFunctionCompartment() signals if activity occurred at a certain time index. Root types will diplay activity.
		if (State.isRoot(this) && State.isParentRoot(this) == false)
		{
			return true;
		}
		if (State.isRoot(this) && State.isParentRoot(this) == false && State.isTrainableBasicChild(this))
		{
			return false;//WHY???????
		}
		if (State.isParentRoot(this) == false && State.isTrainableBasicChild(this))
		{
			return true;
		}
		if(Hi.compareTo(BigInteger.ONE)==0) 
		{
			if(isDerivable()) {return true;}
		}
		NVMemory Memory = (NVMemory)findModule(NVMemory.class);
		if (Memory != null) 
		{
			return Memory.getActivityAt(Hi);
		}
		return false;
	}
	public double[] ActivationAt(BigInteger Hi) 
	{
		if(Hi.compareTo(BigInteger.ONE)==1) 
		{
			if(hasModule(NVMemory.class)) 
			{
				NVMemory Memory = (NVMemory)findModule(NVMemory.class); 
			    if(Memory.hasDataMemory())
			    {
			    	return Memory.getDataAt(Memory.activeTimeDeltaWithin(Hi)).getActivation();
			    }
			}
		}
		return getActivation();
	}
	// DISPLAY CorefindFunctionCompartment():
	// ============================================================================================================================================================================================
	public synchronized void displayCurrentState() 
	{
		shoutLine("NVertex->displayCurrentState():");
		shoutLine("========================================================================");
		shoutLine(toString());
		shout("\n");
		shoutLine("-displayCurrentState|:\n");
		if(State.isRoot(this)) 
		{
			shoutLine("-displayCurrentState|: Node type:");
			shoutLine("-displayCurrentState|: -------------------------------------");
			shoutLine("-displayCurrentState|: isBasicChildNode: "+State.isBasicChild(this));
			shoutLine("-displayCurrentState|: isParentRootNode: "+State.isParentRoot(this));
			shoutLine("-displayCurrentState|: isMemoryRootNode: "+State.isMemoryChild(this));
			shoutLine("-displayCurrentState|: isTrainableBasicChildNode: "+State.isTrainableBasicChild(this));
			shoutLine("-displayCurrentState|: isRootTipNode: "+State.isRootTip(this));
			shoutLine("-displayCurrentState|: isInputRootNode: "+State.isInputRoot(this));
			shoutLine("-displayCurrentState|: -------------------------------------");
			
			System.out.println("-displayCurrentState|: This neuron has a property hub!");
			System.out.println("-displayCurrentState|: Properties:");
			if(this.hasModule(NVMemory.class))       
			{
			    NVMemory Memory = (NVMemory)this.findModule(NVMemory.class);
			    System.out.println("-displayCurrentState|: has memory: true => hasActivityMemory: "+Memory.hasActivityMemory()+"; hasDataMemory: "+Memory.hasDataMemory()+"; " );
			}
			else
			{   
			    System.out.println("-displayCurrentState|: has memory: false");
			}
			shoutLine("-displayCurrentState|: has control frame: "+this.hasModule(NVControlFrame.class));
			shoutLine("-displayCurrentState|: has gradient optimizer: "+this.hasModule(NVOptimizer.class));
		}
		shoutLine("-displayCurrentState|: ~~~~~~~~~~~~~~~~~~~~~~~~\n");
		shoutLine("-displayCurrentState|: Data[Di].Inputs, Inputs, Derivatives, Weights and Shifts:");
		shoutLine("-displayCurrentState|: ------------------");
		if(findModule(Function.class)==null)
		{
			shoutLine("-displayCurrentState|: Calculation Core is null => Value stored in local value array.");
		}
		else                      
		{
			shoutLine("-displayCurrentState|: Calculation Core exists => Value stored in core.");
		}
		for(int i = 0; i< Neuron.length; i++)
		{
			for (int Ii = 0; Ii < Neuron[i].Input.length; Ii++)
			{
				if(Neuron[i].getInput()==null)
				{
					shoutLine("-displayCurrentState|: LocalData[1] is empty => Value space not instantiated.");
				}
				else 
				{
					shoutLine("-displayCurrentState|: Value[" + Ii + "]:(" + Neuron[i].getInput()[Ii]+ ");");
				}
				if(Neuron[i].getInputDerivative()==null)
				{
					shoutLine("-displayCurrentState|: LocalData[2] is empty => Value deriviation space not instantiated.");
				}
				else 
				{
					 shoutLine("-displayCurrentState|: dInput[" + Ii + "]:(" + Neuron[i].getInputDerivative()[Ii] + ");");
				}
				if(Neuron[i].Input[Ii].Bias==null)
				{
					shoutLine("-displayCurrentState|: Bias is empty => Bias space not instantiated.");
				}
				else
				{
					  shoutLine("-displayCurrentState|: Bias[" + Ii + "]:(" + Neuron[i].Input[Ii].Bias + ");");
				}
				if (Connection[Ii] != null) 
				{
					shoutLine("-displayCurrentState|: AIC[" + Ii + "][length:(" + Connection[Ii].length + ")]");
				} 
				else 
				{
					shoutLine("-displayCurrentState|: AIC[" + Ii + "]==null!");
				}
				if (!isFirst()) 
				{
					if(Connection != null) 
					{
						if(Connection[Ii]!=null) 
						{
						    shout("-displayCurrentState|: Weight[" + Ii + "][0-" + (Connection[Ii].length - 1) + "]:( ", true);
						    for (int Ni = 0; Ni < Connection[Ii].length; Ni++) 
						    {
							      shout("[" + Ni + "]:(" + Connection[Ii][Ni].size() + "), ");
						    }
						}
						else
						{shoutLine("-displayCurrentState|: Weight["+Ii+"]==null!");}
					}
					shout(");\n");
				}
			}
			shout("\n");
		}	
	}
	protected void shoutLine(String message) {shoutLine(message, true);}

	protected void shoutLine(String message, boolean addName) 
	{
		if(addName) 
		{
			System.out.println("|NEURON-[" + ID + "]|: " + message + " ");
		} 
		else 
		{
			System.out.println(message + " ");
		}
		NVControlFrame controlFrame = (NVControlFrame)findModule(NVControlFrame.class);
		if (controlFrame != null) 
		{
			controlFrame.println(message);
		}
	}
	protected void shout(String message){shout(message, false);}

	private void shout(String message, boolean addName) 
	{
		if(addName)
		{
			System.out.print("|NEURON-[" + ID + "]|: " + message);
		} 
		else 
		{
			System.out.print(message);
		}
		NVControlFrame controlFrame = (NVControlFrame)findModule(NVControlFrame.class);
		if (controlFrame != null) 
		{
			controlFrame.print(message);
		}
	}
	// asCore
	// =============================================================================================================================================================================================
	@Override // NVNode
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public NVertex asCore() 
	{
		return this;
	}
	@Override// NVNode
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void copy(NVertex other) //TODO: write internalize test cases!
	{
		this.setInstantGradientApply(other.isInstantGradientApply());
		this.addModule((Function)other.findModule(Function.class));//Warningg!!!!
		Neuron = new NVNeuron[other.asCore().Neuron.length];
		//-----------------------------------------------------------------------------
		for(int Di = 0; Di< Neuron.length; Di++)
    	{
			Neuron[Di] = new NVNeuron(other.asCore().inputSize());
			if(other.isConvergent()) 
			{
				Neuron[Di].setOutput(new double[3]);
			}
			else
			{
				if(other.isDerivable()) 
				{
					Neuron[Di].setOutput(new double[2]);
				}
			}
    	}
		this.setLast(other.isLast());
		this.setHidden(other.isHidden());
		this.setFirst(other.isFirst());
		
		Connection = new NVNode[other.asCore().Connection.length][];
		for(int Vi=0; Vi<this.size(); Vi++) 
    	{
			//Connection[Di] = new NVNode[otherVertex.Core().Neuron[Di].Input.length];
			Neuron[Vi].Input = new NVInput[other.asCore().Neuron[Vi].Input.length];
			
			Neuron[Vi].setOutput(null);
			Neuron[Vi].setInput(null);
			Neuron[Vi].setInputDerivative(null);
			
			if(other.Neuron[0].getOutput()!=null)
			{
				Neuron[Vi].setOutput(new double[other.Neuron[0].getOutput().length]);}
			if(other.Neuron[0].getInput()!=null)
			{
				Neuron[Vi].setInput(new double[other.Neuron[0].getInput().length]);}
			if(other.Neuron[0].getInputDerivative()!=null)
			{
				Neuron[Vi].setInputDerivative(new double[other.Neuron[0].getInputDerivative().length]);}
			
			for(int Ii = 0; Ii< Neuron[Vi].Input.length; Ii++)
			{
				Neuron[Vi].Input[Ii] = new NVInput(other.asCore().Neuron[Vi].Input[Ii].size(), true);
				Neuron[Vi].Input[Ii].Bias = other.asCore().Neuron[Vi].Input[Ii].Bias;
				Neuron[Vi].Input[Ii].BiasGradient = other.asCore().Neuron[Vi].Input[Ii].BiasGradient;
				
				if(other.asCore().Neuron[Vi].Input[Ii].Weight!=null)
				{
					Neuron[Vi].Input[Ii].Weight = new double[other.asCore().Neuron[Vi].Input[Ii].Weight.length];
				}
				if(other.asCore().Neuron[Vi].Input[Ii].WeightGradient!=null)
				{
					Neuron[Vi].Input[Ii].WeightGradient = new double[other.asCore().Neuron[Vi].Input[Ii].WeightGradient.length];
				}
			}
			this.setBiasTrainable(other.biasIsTrainable());
			this.setWeightTrainable(other.weightIsTrainable());
			this.ID=other.ID; 
			other.ID++;
    	}
		this.setParalyzed(other.isParalyzed());
		this.setGradientSumming(other.isGradientSumming());
		
		if(other.asRoot()!=null && this.asRoot()!=null)
		{
			NVParent superCore = other.asRoot().asParent();
		    this.asRoot().asChild().setParent(superCore);
		}
	}

	@Override
	public long getActivitySignal() {
		return 1;
	}

	@Override
	public boolean isExecutable() {
		return true;
	}
	@Override
	public boolean isRoot() {
		return false;
	}
	@Override
	public boolean isNeuron()
	{
		return false;
	}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//NVertex implementation ends here!

    public static class State // Contains utililty functions used to determine the state of the node.
    {//=============================================================================================
    	//-------------------------------------------------------------------
		public static boolean isRoot(NVNode node) 
		{
			if(node.asCore() instanceof NVertex_Root) 
			{
				return true;
			}
			return false;
		}
		//-------------------------------------------------------------------
		public static boolean isParentRoot(NVNode unit)
		{
			if (isRoot(unit.asCore())) 
			{
				if(unit.asCore().asRoot()!=null && unit.asCore().asRoot().asParent()!=null) 
				{
					return true;
				}
			}
			return false;
		}
		//-------------------------------------------------------------------
		public static boolean isInputRoot(NVNode node) 
		{
			boolean soilFound = false;
			if(node.asCore().Neuron[0].Input !=null)
			{
				for(int Ii=0; Ii<node.asCore().inputSize(); Ii++) 
				{
					if(node.asCore().Connection[Ii]!=null)
					{
						if(State.isParentRoot(node.asCore().Connection[Ii][0].asCore())
						 ||State.isRoot(node.asCore().Connection[Ii][0].asCore())==false) 
						{
							soilFound=true;
						}
					}
					if(soilFound) 
					{
						break;
					}
				}
			}
			if (isRoot(node) && soilFound) 
			{
				return true;
			}	
			return false;
		}
		//=================================================================================
		public static boolean isMemoryChild(NVNode unit)
		{
			NVMemory M = (NVMemory) unit.asCore().findModule(NVMemory.class);
			if(M!=null)
			{
				if (isRoot(unit) == true) 
				{
					if (M.hasDataMemory() == true && M.hasActivityMemory() == false)
					{
						return true;
					}
				}
			}
			return false;
		}
		//-----------------------------------------------------------------------
		public static void turnIntoMemoryChild(NVNode node, NVNode parent)
		{
			if(isRoot(node)==false && isRoot(parent)==false)
			{
				return;
			}
			turnIntoChildOfParent(node, parent);
			if(node.asCore().hasModule(NVMemory.class)==false)
			{
				node.asCore().addModule(new NVMemory(3));
			}
			NVMemory Memory = (NVMemory)node.asCore().findModule(NVMemory.class);
			Memory.forgetActivity();
		}
		//-----------------------------------------------------------------------
		public static boolean isTrainableBasicChild(NVNode unit)
		{
			if (isRoot(unit)==false) 
			{
				return false;
			}
			if (State.isMemoryChild(unit.asCore()))
			{
				return false;
			}
			if (State.isBasicChild(unit.asCore())) 
			{
				if (unit.asCore().weightIsTrainable() == true || unit.asCore().biasIsTrainable() == true) 
				{
					return true;
				}
				return false;
			}
			NVMemory Memory = (NVMemory)unit.asCore().findModule(NVMemory.class);
			if (Memory == null) 
			{
				if (unit.asCore().weightIsTrainable() == true || unit.asCore().biasIsTrainable() == true) 
				{
					return true;
				}
				return false;
			}
			if (Memory.hasDataMemory() == false)
			{
				if(Memory.getDataAt(0).hasInputDerivative(0))
				{
					if (unit.asCore().weightIsTrainable() == true || unit.asCore().biasIsTrainable() == true)
					{
						return true;
					}
				}
				return false;
			}
			return false;
		}
		//-----------------------------------------------------------------------
		public static boolean isRootTip(NVNode node) 
		{
			if((isRoot(node))) 
			{
				if(isParentRoot(node)==false && (isInputRoot(node) || isTrainableBasicChild(node) || isMemoryChild(node)))
				{
					return true;
				}
			}
			return false;
		}
		//-----------------------------------------------------------------------
		public static void turnIntoParentRoot(NVNode node) 
		{
			if(isRoot(node) && isChild(node)) 
			{
				((NVChild)node)
				.setParent
				(
					(NVParent)node
				);
			}
		}
		//-----------------------------------------------------------------------
		public static boolean isChild(NVNode unit) 
		{
			if(isParentRoot(unit) || unit.asCore().asRoot()==null)
			{
				return false;
			}
			return true;
		}
		//-----------------------------------------------------------------------
		public static boolean turnIntoChildOfParent(NVNode subUnit, NVNode superUnit) 
		{
			if(isRoot(subUnit)==false && isRoot(superUnit)==false) 
			{
				return false;
			}
			((NVChild)subUnit)
			.setParent
			(
				superUnit.asRoot().asParent()
			);
			return true;
		}
		//-----------------------------------------------------------------------
		public static boolean isBasicChild(NVNode unit) 
		{
			if(isRoot(unit) && !isParentRoot(unit) && !isMemoryChild(unit))
			{
				return true;
			}
			return false;
		}
		//-----------------------------------------------------------------------
		public static boolean turnIntoBasicChild(NVNode node, NVNode superUnit) 
		{
			if(isRoot(node)==false || isRoot(superUnit)==false) 
			{
				return false;
			}
			if(turnIntoChildOfParent(node, superUnit)==false)
			{
				return false;
			}
			node.asCore().shoutLine("-turnIntoBasicRootUnit()|: Property hub is not null! -> looking for panel compartment!");
			node.asCore().removeModule(NVMemory.class);
			//node.asCore().removeModule(NTimeoutCompartment.class);
			
			if(!node.asCore().biasIsTrainable() && !node.asCore().weightIsTrainable()) 
			{
				node.asCore().removeModule(NVOptimizer.class);
			}
			//NVCloak Cloak = (NVCloak)node.asCore().find(NVCloak.class);
			//if(Cloak!= null)
			//{
			//	Cloak.abandonYourself();
			//}
			return true;
		}
		//-----------------------------------------------------------------------
		public static boolean isTrainableNeuron(NVNode unit) 
		{
			if(isRoot(unit)) 
			{
				return false;
			}
			if(unit.asCore().findModule(NVMemory.class)==null) 
			{
				return false;
			}
			return true;//CHECK IF MEMORY WORKS FINE.... has this and that
		}
		//-----------------------------------------------------------------------
		public static boolean turnIntoTrainableNeuron(NVNode unit) 
		{
			if(isRoot(unit)) 
			{
				return false;
			}
			if(unit.asCore().findModule(NVMemory.class)==null) 
			{
				unit.asCore().addModule(new NVMemory(3));
			}
			return true;
		}
		//-----------------------------------------------------------------------
    
    }//=============================================================================================

}
