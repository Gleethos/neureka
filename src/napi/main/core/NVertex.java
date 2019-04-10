
package napi.main.core;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Random;

import napi.main.core.calc.NVFHead;
import napi.main.core.calc.NVFunction;
import napi.main.core.comp.NVData;
import napi.main.core.comp.NVElement;
import napi.main.core.comp.NVElementVariable;
import napi.main.core.imp.NVertex_Root;
import napi.main.core.mem.NVMemory;
import napi.main.core.opti.NVOptimizer;
import napi.main.exec.NVExecutable;
import ngui.NVControlFrame;

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

	protected NVNode[][] Connection; // [Input size][connections]
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
	public int size() {return Element.length;}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NVNode
	public void apply(NAction Actor) {Actor.act(this);}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NVNode
	public NVNode asNode(){if((findModule(NVCloak.class))!=null) {return ((NVCloak)findModule(NVCloak.class));}return this;}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NVNode
	public NVRoot asRoot() {if(this instanceof NVRoot) {return (NVRoot)this;}return null;}
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override //NCore_Neuron
	public NVNeuron asNeuron() {if(this instanceof NVNeuron) {return (NVNeuron)this;}return null;}
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
		Element = new NVElement[size];
		
		for(int Vi=0; Vi<size; Vi++) 
		{
			Element[Vi] = new NVElement(1);
			Element[Vi].Variable = new NVElementVariable[1];
			Element[Vi].Variable[0] = new NVElementVariable(0, derivable);
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
		NVFunction newFunction = new NVFHead();
		switch(f_id) 
		{   
		    case 0:  newFunction = newFunction.newBuild("prod(relu(Ij))"); break;
			case 1:  newFunction = newFunction.newBuild("prod(sig(Ij))"); break;
			case 2:  newFunction = newFunction.newBuild("prod(tanh(Ij))"); break;
			case 3:  newFunction = newFunction.newBuild("prod(quadr(Ij))"); break;
			case 4:  newFunction = newFunction.newBuild("prod(lig(Ij))"); break;
			case 5:  newFunction = newFunction.newBuild("prod(lin(Ij))"); break;
			case 6:  newFunction = newFunction.newBuild("prod(gaus(Ij))"); break;
			default: newFunction = newFunction.newBuild("prod(tanh(Ij))");break;
		}
		addModule((NVFunction)newFunction);
		if(delay>0) {addModule(new NVMemory(delay));}
		setDerivable(derivable);
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
			if(Modules.get(Mi) instanceof NVFunction){
				if(this.hasModule(NVFunction.class)) {info = ((NVFunction) findModule(NVFunction.class)).expression();}
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
		if (Element[0].Variable.length >= targIi) 
		{
			return;//Target index does not exist.
		}
		for(int Vi=0; Vi<Element.length; Vi++)
		{
			Element[Vi].removeInputAt(targIi);
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
			addModule(new NVOptimizer(Element.length, Connection, GradientOptimizer.getOptimizationID()));
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
		if (Element[0].Variable.length < targIi)
		{
			return;
		}
		for(int Di=0; Di<Element.length; Di++)
		{
			Element[Di].addInputAt(targIi);
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
			addModule(new NVOptimizer(Element.length, Connection, GradientOptimizer.getOptimizationID()));
		}
		NVMemory Memory = (NVMemory)findModule(NVMemory.class);
		if(Memory!=null) {Memory.resetMemory();}
	}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public void swapConnection(NVNode former, NVNode replacement) 
	{
		for(int Di=0; Di<Element.length; Di++)
		{
			former = former.asNode(); replacement = replacement.asNode();
			if(Element[Di].Variable==null) {return;}
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
		for (int Ii = 0; Ii < Element[0].Variable.length; Ii++) 
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
			shoutLine("-disconnect(...)|: AIC[targIi:(" + targIi + ")] is null, empty input Data[Di].Input !"); 
			return;
		}
		int i = - 1;
		for(int Ni=0; Ni<Connection[targIi].length; Ni++) 
		{
			Connection[targIi][Ni]=Connection[targIi][Ni].asNode();
		    if(Connection[targIi][Ni].asCore()==ConnectionNode.asCore()) {i=Ni;}
		}
		if(i==-1) {return;}
		shoutLine("-disconnect(...)|: removing Data[Di].Input now! |: ->AIC.length:(" + inputSize()
				+ ") |: calling uptdateInputData[Di].Input(Ni:(" + i + "), remove:(false)); ");
		
		NVUtility<NVNode> utility1 = new NVUtility<NVNode>();
		Connection[targIi] = utility1.updateArray(Connection[targIi], i, true, NVNode.class);
		
		for(int Di=0; Di<Element.length; Di++)
		{
			NVUtility<double[]> Utility2 = new NVUtility<double[]>();
			for(int Wi=0; Wi<ConnectionNode.size(); Wi++) 
			{
			    Element[Di].Variable[targIi].Weight = Utility2.updateArray(Element[Di].Variable[targIi].Weight, i, true);
			}
		}
		if (Connection[targIi] != null) 
		{shoutLine("-disconnect(...)|: new AIC -> AIC-I[" + targIi + "].length = " + Connection[targIi].length);}
		NVOptimizer oldGradientOptimizer = (NVOptimizer) findModule(NVOptimizer.class);
		if (oldGradientOptimizer != null) 
		{
			removeModule(NVOptimizer.class);
			addModule(new NVOptimizer(Element.length, Connection, oldGradientOptimizer.getOptimizationID()));
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
		for(int Ii = 0; Ii < Element[0].Variable.length; Ii++) 
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
		shoutLine("NVertex->connect("+node.getClass()+" node, int " +targIi +")|: Connecting to 'Input-Index':" + targIi + "  !");
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
		shoutLine("-connect(..., int 'targIi':"+targIi+")|: for(int Vi=0; Vi<Element.length; Vi++): ...");
		if (Element[0].Variable == null)
		{
			shoutLine("-connect(..., int 'targIi':"+targIi+")|: (Element[0].Dendrite == null):");
			if(targIi==0) 
			{ 
				shoutLine("-connect(..., int 'targIi':"+targIi+")|: =>(targIi==0):");
				shoutLine("-connect(..., int 'targIi':"+targIi+")|: =>=> Element[Vi].Input = new NVElementWeightContext[1];");
				shoutLine("-connect(..., int 'targIi':"+targIi+")|: =>=> Connection = new NVNode[1][];") ;
				Element[0].Variable = new NVElementVariable[1];
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
			shoutLine("-connect(..., int 'targIi':"+targIi+")|: (Element[Vi].Dendrite != null):");
		}
		if (targIi >= Element[0].Variable.length) 
		{
			shoutLine("-connect(..., int 'targIi':"+targIi+")|: (targIi >= Element[Vi].Dendrite.length):");
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
		if (Element[0].Variable[targIi] == null) 
		{
			shoutLine("-connect(..., int 'targIi':"+targIi+")|: AIC[targIi:(" + targIi + ")] is null, currently empty input Data[Di].Input !");
		}
		int i = 0;
		if(Connection[targIi]!=null) 
		{
			i=Connection[targIi].length;
		}
		shoutLine("-connect(..., int 'targIi':"+targIi+")|: Connecting now! | calling uptdateInputData[Di].Input(Ni:(" + i + "), remove:(false)); ");
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
		if(Element[0].Variable!=null) 
		{
			int Wi=0;
			for(int Ni=0; Ni<Connection[targIi].length; Ni++) 
			{
				Wi+=Connection[targIi][Ni].size();
			}
			for(int wi=0; wi<Wi; wi++) 
			{
				NVUtility<double[]> utility2 = new NVUtility<double[]>();
				Element[0].Variable[targIi].Weight =  utility2.updateArray(Element[0].Variable[targIi].Weight, i, false);
			}
		}
		if(this.is(NVertex.Root) && node.asCore().is(NVertex.Child) && Connection[targIi].length==1) 
		{
			Element[0].Variable[targIi].Weight = null;
		}
		shoutLine("-connect(..., int 'targIi':"+targIi+")|: 'Connection[targIi].length':" + Connection[targIi].length);
		NVOptimizer GradientOptimizer = (NVOptimizer)findModule(NVOptimizer.class);
		if (GradientOptimizer != null)
		{
			shoutLine("-connect(..., int 'targIi':"+targIi+")|: (GradientOptimizer != null):");
			removeModule(NVOptimizer.class);
			addModule(new NVOptimizer(Element.length, Connection, GradientOptimizer.getOptimizationID()));
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
		for(int Vi=0; Vi<Element.length; Vi++)
		{
			if (Element[Vi].Variable == null)
			{
				shoutLine("-randomizeShiftAndWeightMatrix|: Weight randomization failed! No weight to randomize.");
			}
			else
			{
				for (int Ii = 0; Ii < Element[Vi].Variable.length; Ii++)
				{
					if (Element[Vi].Variable[Ii].Weight != null)
					{
						for (int Wi = 0; Wi < Element[Vi].Variable[Ii].Weight.length; Wi++)
						{
							Element[Vi].Variable[Ii].Weight[Wi] = NVUtility.getDoubleOf(Wi+Ii+Vi*Ii*Vi);
						}
						Element[Vi].Variable[Ii].Bias = NVUtility.getDoubleOf(278+Ii+Vi*Ii*Vi);
					}
				}
			}
			setError(Vi,0);
			NVCloak cloak = (NVCloak)findModule(NVCloak.class);
			if(cloak!=null)
			{
				cloak.setError(Vi, 0.0);
			}
		}
	}
	// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	private void instantiateWeightMatrix()
	{
		for(int Di=0; Di<Element.length; Di++)
		{
			for (int Ii = 0; Ii < Element[Di].Variable.length; Ii++)
			{
				if (Element[Di].Variable[Ii] != null)
				{
					int size = 0;
					for(int Ni=0; Ni<inputSize(); Ni++) {size += Connection[Ii][Ni].size();}
					Element[Di].Variable[Ii].Weight = new double[size];
				}
			}
		}
	}
	// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public void biasAll()
	{
		for(int Di=0; Di<Element.length; Di++)
		{
			for (int Ii = 0; Ii < Element[Di].Variable.length; Ii++)
			{
				Element[Di].Variable[Ii].Bias = 0.0;
			}
		}
	}
	// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	protected void instantiateWeightGradient()
	{
		for(int Di=0; Di<Element.length; Di++)
		{
			for (int Ii = 0; Ii < Element[Di].Variable.length; Ii++)
			{
				if (Element[Di].Variable[Ii] != null)
				{
					if(Element[Di].Variable[Ii].Weight!=null)
					{
						Element[Di].Variable[Ii].WeightGradient = new double[Element[Di].Variable[Ii].Weight.length];
					}
				}
			}
		}
	}
	// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	protected void instantiateBiasGradient()
	{
		for(int Di=0; Di<Element.length; Di++)
		{
			for (int Ii = 0; Ii < Element[Di].Variable.length; Ii++)
			{
				Element[Di].Variable[Ii].BiasGradient = 0.0;
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
		for(int Vi=0; Vi<Element.length; Vi++)
		{
			if(!isFirst())
			{
				Element[Vi].setInput(null);
			}
			NVOptimizer GradientOptimizer = (NVOptimizer)findModule(NVOptimizer.class);
			for (int Ii = 0; Ii < Element[Vi].Variable.length; Ii++)
			{
				if (Element[Vi].Variable[Ii].Weight != null && weightIsTrainable())
				{
					int Wi=0;
					for (int Ni = 0; Ni < Connection[Ii].length; Ni++)
					{
						if (GradientOptimizer != null)
						{
							int limit = Connection[Ii][Ni].size()+Wi;
	        		   		while(Wi<limit)
	        		   		{
	        		   			double optimized = GradientOptimizer.getOptimized(Element[Vi].Variable[Ii].Weight[Wi], Ii, Wi);
	        		   			shoutLine("-applyGradients|: Applying optimized WG[i" + Ii + "][n" + Ni + "]:("
								+ Element[Vi].Variable[Ii].WeightGradient[Wi] + "=>" + optimized + ") to W[i" + Ii + "][n" + Ni + "]:("
								+ Element[Vi].Variable[Ii].Weight[Wi] + ");");
	        		   			//======================================
	        		   			Element[Vi].Variable[Ii].Weight[Wi] += optimized;
	        		   			//======================================
	        		   			Wi++;
			     			}
						}
						else
						{
							int limit = Connection[Ii][Ni].size()+Wi;
	        		   		while(Wi<limit)
	        		   		{
	        		   			shoutLine("-applyGradients|: Applying WG[i" + Ii + "][" + Wi + "]:(" + Element[Vi].Variable[Ii].WeightGradient[Wi]
	        		   				+ ") to W[i" + Ii + "][n" + Wi + "]:(" + Element[Vi].Variable[Ii].Weight[Wi] + ");");
	        		   			//======================================
	        		   			Element[Vi].Variable[Ii].Weight[Wi] += Element[Vi].Variable[Ii].WeightGradient[Wi];
	        		   			//======================================
	        		   			Wi++;
			     			}
						}
					}
				}
				// SHIFT GRADIENTS:
				if (Element[Vi].Variable[Ii].Bias != null && biasIsTrainable())
				{
					if (GradientOptimizer != null)
					{//======================================
						Element[Vi].Variable[Ii].Bias += GradientOptimizer.getOptimized(Element[Vi].Variable[Ii].BiasGradient, Vi, Ii);
					}//======================================
					else
					{//======================================
						Element[Vi].Variable[Ii].Bias += Element[Vi].Variable[Ii].BiasGradient;
					}//======================================
				}
			}
		}
		nullGradients();
	}

	public void nullGradients(){for(int Di=0; Di<Element.length; Di++) {Element[Di].nullGradients();}}

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
	public synchronized NVFunction getFunction() {return (NVFunction)findModule(NVFunction.class);}
	public synchronized void 	    setFunction(NVFunction function) {addModule(function);}
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
			    if(Memory.hasActivationMemory()) 
			    {
			    	return Memory.getActivationAt(Memory.activeTimeDeltaWithin(Hi))[0];
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
			    System.out.println("-displayCurrentState|: has memory: true => hasActivityMemory: "+Memory.hasActivityMemory()+"; hasDeriviationMemory: "+Memory.hasDeriviationMemory()+"; hasActivationMemory: "+Memory.hasActivationMemory()+";" );
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
		if(findModule(NVFunction.class)==null) 
		{
			shoutLine("-displayCurrentState|: Calculation Core is null => Input stored in local data array.");
		}
		else                      
		{
			shoutLine("-displayCurrentState|: Calculation Core exists => Input stored in core.");
		}
		for(int i=0; i<Element.length; i++) 
		{
			for (int Ii = 0; Ii < Element[i].Variable.length; Ii++) 
			{
				if(Element[i].getInput()==null) 
				{
					shoutLine("-displayCurrentState|: LocalData[1] is empty => Input space not instantiated.");
				}
				else 
				{
					shoutLine("-displayCurrentState|: Input[" + Ii + "]:(" + Element[i].getInput()[Ii]+ ");");
				}
				if(Element[i].getInputDerivative()==null) 
				{
					shoutLine("-displayCurrentState|: LocalData[2] is empty => Input deriviation space not instantiated.");
				}
				else 
				{
					 shoutLine("-displayCurrentState|: dInput[" + Ii + "]:(" + Element[i].getInputDerivative()[Ii] + ");");
				}
				if(Element[i].Variable[Ii].Bias==null) 
				{
					shoutLine("-displayCurrentState|: Bias is empty => Bias space not instantiated.");
				}
				else
				{
					  shoutLine("-displayCurrentState|: Bias[" + Ii + "]:(" + Element[i].Variable[Ii].Bias + ");");
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
	//asCore
	public void setDerivable(boolean derivable) 
	{
		NVCloak cloak = (NVCloak)findModule(NVCloak.class);
		double[][][] oldData = new double[this.size()][][]; 
		for(int Vi=0; Vi<Element.length; Vi++) 
		{
			oldData[Vi] = new double[3][];
			oldData[Vi][0] = Element[Vi].getOutput();//Saving data.
			oldData[Vi][1] = Element[Vi].getInput();//Saving data.
			oldData[Vi][2] = Element[Vi].getInputDerivative();//Saving data.
					
		}
		//------------------------------------------------------------------
		if(derivable) 
		{
			if(this.isDerivable())//Already derivable! -> exposing result (cloak)!
			{
				if(cloak!=null) 
		        {
				    double[][] data = new double[Element.length][];
					for(int Di=0; Di<data.length; Di++) 
					{
						data[Di] = Element[Di].getOutput();
					}
			        cloak.setPublicData(data);
		        }
		        return;
			}
			else
			{
			   	for(int Di=0; Di<Element.length; Di++) 
			   	{
			   		Element[Di].setOutput(oldData[Di][0]);
			   		Element[Di].setInput(oldData[Di][1]);
			   		Element[Di].setInputDerivative(oldData[Di][2]);
			   	}			    	
			}
			if(oldData[0]==null) 
			{
				if(Connection==null) 
				{
					Connection = new NVNode[1][];
				}
				for(int Vi=0; Vi<Element.length; Vi++) 
		    	{
					Element[Vi].setInput(new double[this.inputSize()]);
		    	}
		    }
			for(int Di=0; Di<Element.length; Di++) 
	    	{
				Element[Di].setInputDerivative(new double[this.inputSize()]);
	    	}
			if(oldData[0]!=null) 
		    {
				if(oldData[0][0]!=null) 
				{
					for(int Di=0; Di<Element.length; Di++) 
				   	{
						Element[Di].setOutput(oldData[Di][0]);
			    	}
				}
				else 
				{
					for(int Di=0; Di<Element.length; Di++) 
			    	{
						Element[Di].setOutput(new double[2]);
			    	}
				}
			}
			else
			{
				for(int Di=0; Di<Element.length; Di++) 
		    	{
					Element[Di].setOutput(new double[2]);
		    	}
			}
			for(int Di=0; Di<Element.length; Di++) 
	    	{
				shoutLine("-setDerivable(true)|: New Output has been set! Output size: "+Element[Di].getOutput().length);
	    	}
		}
		else 
		{
			if(this.isDerivable() == false)// Element[0].Data.length==2 && Element[0].Data[0].length==1
			{	
				double[][] newPublic = new double[Element.length][]; 
				for(int Di=0; Di<Element.length; Di++) 
			    {	
					newPublic[Di] = Element[Di].getOutput();
			    }	
				if(cloak!=null) {cloak.setPublicData(newPublic);}
				return;
			}
			for(int Di=0; Di<Element.length; Di++) 
	    	{	
				Element[Di].setOutput(new double[2]);
				Element[Di].setInput(new double[this.inputSize()]);
				Element[Di].setInputDerivative(null);
				shoutLine("-setDerivable(false)|: New LocalData has been set!");
	    	}
		}	
		double[][] newPublic = new double[Element.length][];
		
		for(int Vi=0; Vi<Element.length; Vi++) 
	   	{
			if(oldData[Vi]==null) 
			{
				newPublic[Vi] = Element[Vi].getOutput();
				if(cloak!=null) 
				{
					cloak.setPublicData(newPublic);
				} 
				return;
		    }
			else
			{
				Element[Vi].setOutput(oldData[Vi][0]);
				Element[Vi].setInput(oldData[Vi][1]);
				Element[Vi].setInputDerivative(oldData[Vi][2]);	
			}
    	}
		if(cloak!=null) 
		{
			for(int Di=0; Di<Element.length; Di++) 
	    	{	
				newPublic[Di] = Element[Di].getOutput();
	    	}
			cloak.setPublicData(newPublic);			
		}
	}
	// asCore
	public synchronized void setConvergence(boolean converge) 
	{
		// Adds entry in the Element[i].Data[0] array for a value called 'optimum'. 
	    // The node will converge towards this value (either by itself or indirectly (loss node)). 
	    // Last layer nodes converge with the Help of a loss node. Non last layer nodes converge without help!
		if(this.isDerivable()==false) 
		{
			setDerivable(true);
		}
		for(int Di=0; Di<Element.length; Di++) 
    	{	
			if(converge==false) {return;}
			Element[Di].setInputDerivative(new double[this.inputSize()]);
			
			if(Element[Di].getOutput()==null) 
			{
				if(converge) 
				{
					Element[Di].setOutput(new double[3]);
				}
				else
				{
					Element[Di].setOutput(new double[2]);
				}
			}
			if(Element[Di].getOutput().length != 3) 
			{
				double[] oldData = Element[Di].getOutput(); 
				Element[Di].setOutput(new double[3]);
				Element[Di].setActivation(oldData[0]);;
				Element[Di].setError(oldData[1]); //.getOutput()[1]=;
			}
    	}
		double[][] newPublic = new double[Element.length][];
		
		NVCloak Public = (NVCloak)findModule(NVCloak.class);
		if(Public!=null) 
		{
			for(int Di=0; Di<Element.length; Di++) 
	    	{	
				newPublic[Di] = Element[Di].getOutput();
	    	}
			Public.setPublicData(newPublic);
		}
	}
	// =============================================================================================================================================================================================
	@Override // NVNode
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public NVertex asCore() 
	{
		return this;
	}
	@Override// NVNode
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void copy(NVertex other) 
	{
		this.setInstantGradientApply(other.isInstantGradientApply());
		this.addModule((NVFunction)other.findModule(NVFunction.class));//Warningg!!!!
		Element = new NVElement[other.asCore().Element.length];
		//-----------------------------------------------------------------------------
		for(int Di=0; Di<Element.length; Di++) 
    	{
			Element[Di] = new NVElement(other.asCore().inputSize());
			if(other.isConvergent()) 
			{
				Element[Di].setOutput(new double[3]);
			}
			else
			{
				if(other.isDerivable()) 
				{
					Element[Di].setOutput(new double[2]);
				}
			}
    	}
		this.setLast(other.isLast());
		this.setHidden(other.isHidden());
		this.setFirst(other.isFirst());
		
		Connection = new NVNode[other.asCore().Connection.length][];
		for(int Vi=0; Vi<this.size(); Vi++) 
    	{
			//Connection[Di] = new NVNode[otherVertex.Core().Element[Di].Variable.length];		
			Element[Vi].Variable = new NVElementVariable[other.asCore().Element[Vi].Variable.length];
			
			Element[Vi].setOutput(null);
			Element[Vi].setInput(null);
			Element[Vi].setInputDerivative(null);
			
			if(other.Element[0].getOutput()!=null) 
			{Element[Vi].setOutput(new double[other.Element[0].getOutput().length]);}
			if(other.Element[0].getInput()!=null) 
			{Element[Vi].setInput(new double[other.Element[0].getInput().length]);}
			if(other.Element[0].getInputDerivative()!=null) 
			{Element[Vi].setInputDerivative(new double[other.Element[0].getInputDerivative().length]);}
			
			for(int Ii=0; Ii<Element[Vi].Variable.length; Ii++)
			{
				Element[Vi].Variable[Ii] = new NVElementVariable(other.asCore().Element[Vi].Variable[Ii].size(), true);
				Element[Vi].Variable[Ii].Bias = other.asCore().Element[Vi].Variable[Ii].Bias;
				Element[Vi].Variable[Ii].BiasGradient = other.asCore().Element[Vi].Variable[Ii].BiasGradient;
				
				if(other.asCore().Element[Vi].Variable[Ii].Weight!=null) 
				{
					Element[Vi].Variable[Ii].Weight = new double[other.asCore().Element[Vi].Variable[Ii].Weight.length];
				}
				if(other.asCore().Element[Vi].Variable[Ii].WeightGradient!=null) 
				{
					Element[Vi].Variable[Ii].WeightGradient = new double[other.asCore().Element[Vi].Variable[Ii].WeightGradient.length];
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
			if(node.asCore().Element[0].Variable!=null) 
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
					if (M.hasDeriviationMemory() == true && M.hasActivityMemory() == false) 
					{
						return true;
					}
				}
			}
			return false;
		}
		//-----------------------------------------------------------------------
		public static void turnIntoMemoryRoot(NVNode node, NVNode superNode) 
		{
			if(isRoot(node)==false && isRoot(superNode)==false) 
			{
				return;
			}
			turnIntoChildOfParent(node, superNode);
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
			if (Memory.hasDeriviationMemory() == false) 
			{
				if (unit.asCore().weightIsTrainable() == true || unit.asCore().biasIsTrainable() == true) 
				{
					return true;
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
			NVCloak Cloak = (NVCloak)node.asCore().findModule(NVCloak.class);
			if(Cloak!= null)
			{
				Cloak.abandonYourself();
			}
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
