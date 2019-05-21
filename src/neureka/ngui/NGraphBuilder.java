package neureka.ngui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.util.ArrayList;
import java.util.LinkedList;

import neureka.main.core.NVNode;
import neureka.main.core.NVertex;
import neureka.main.core.imp.NVertex_Root;
import neureka.main.exec.cpu.NThreadPool;

public class NGraphBuilder {

	private Thread Worker;
	private NThreadPool ThreadCoordinator;
	NPanel Surface;
	
	private int pressRadius = 100;
	
	public NPanelNode   MouseAttachedPanelNeuron;
    public NPanelNode   ConnectionPanelNeuron;
   
    public int InputIndex = 0;
    
    public NPanelNodeInput ChosenInputNode;

    
	private NCircleMenu MouseAttachedCircleMenu;
	
	NCircleMenu newMenu;
	private NVertex BlueprintNode;
	
	//----------------------------------------------------------------
	public void addMenu(NCircleMenu menu) 
 	{
	 	if(Surface.getMap()==null) {Surface.setMap(new NMap(menu.getX(), menu.getY(), menu.getRadius()*10));}
	 	Surface.setMap(Surface.getMap().addAndUpdate(menu));
 	}
	public NVertex getBlueprintNeuron() 
 	{
 		return BlueprintNode;
 	}
	public NPanel getSurface() 
	{
		return Surface;
	}
	private void addPanelNeuron(int x, int y)       
 	{
 		NPanelNode Unit = new NPanelNode(new NVertex_Root(BlueprintNode), Surface.realX((double)x),Surface.realY((double)y));
 		addPanelNeuron(Unit);
 		Unit.setHasMoved(true);
 	}
 	private void addPanelNeuron(neureka.ngui.NPanelNode PanelNeuron)
 	{
 		if(Surface.getMap()==null) 
 		{
 			Surface.setMap(new NMap(PanelNeuron.getX(),PanelNeuron.getY(),10000));
 		}
 		Surface.setMap(Surface.getMap().addAndUpdate(PanelNeuron));
 	}
	public NGraphBuilder() 
	{	
		Surface = new NPanel();
		NPanelNode Neuron = new neureka.ngui.NPanelNode(-400, 2500, 0);
	 	NVertex.State.turnIntoParentRoot(Neuron.getCore()); // turnIntoSuperRootNode();
	 
	 	addPanelNeuron(Neuron);
	 	NPanelNode Neuron1 = new neureka.ngui.NPanelNode( 700, -2600, 1 );
	 	NVertex.State.turnIntoChildOfParent(Neuron1.getCore(), Neuron.getCore().asCore());//Neuron1.getUnit().turnIntoSubNodeOf(Neuron.getUnit().Core());
	 	NVertex.State.turnIntoBasicChild(Neuron1.getCore(), Neuron.getCore().asCore());//Neuron1.getUnit().turnIntoBasicRootNode();
	 
	 	addPanelNeuron(Neuron1);
	 	Neuron.getCore().connect(Neuron1.getCore());
	 
	 	NPanelNode Neuron4 = new neureka.ngui.NPanelNode( 3090, 890, 2 );
	 	addPanelNeuron(Neuron4);
	 
	 	NVertex.State.turnIntoChildOfParent(Neuron4.getCore(), Neuron.getCore());//Neuron4.getUnit().turnIntoSubNodeOf(Neuron.getUnit().Core());
	 	NVertex.State.turnIntoMemoryChild(Neuron4.getCore(), Neuron.getCore()); // Neuron4.getUnit().turnIntoMemoryRootNode();
	 
		BlueprintNode = new NVertex_Root(1, true, false, 0, 2, 1);
		
		
		
		
		Surface.setPreferredSize(new Dimension(200,30));
		Surface.setBackground(Color.black);
		
		Surface.setPaintAction(
			(surface, brush)->
			{
				brush.setColor(Color.GREEN);
				brush.fillOval((int)surface.realX(surface.lastSenseX())-5, (int)surface.realY(surface.lastSenseY())-5, 10, 10);
					
				if(this.ChosenInputNode!=null)
				{
					double x = this.ChosenInputNode.getX();
				    double y = this.ChosenInputNode.getY();
				    brush.setColor(Color.CYAN);
				    brush.setStroke(new BasicStroke(22));
				    brush.drawLine((int)x, (int)y, (int)surface.realX(surface.lastSenseX()), (int)surface.realY(surface.lastSenseY()));
				}
					
			}
		);
		
		Surface.setClickAction(
			(surface)->
			{
				int[] Click = surface.getClick();
				if(Click==null) {return;}
				int x = Click[0];
				int y = Click[1];
				//checkPanelScaleCenter(getWidth()/2, getHeight()/2);
				//Goal: find something on panel -> top most thing -> attach if is attachable...
				System.out.println("X:"+surface.realX((double)x)+"; Y:"+surface.realY((double)y)+";");
				surface.getListener().setDraggStart(x, y);
				NPanelObject found = surface.findObject(surface.realX(x), surface.realY(y), true, null);
				if(found!=null) 
				{
				    if(found.clickedAt(surface.realX(x), surface.realY(y), surface)) 
				    {	
				    	if(found instanceof NPanelNode) 
				    	{
				    		NPanelNode PU = (NPanelNode)found;		
				    		NPanelNodeInput node = PU.testFor_AndGet_InputNode(surface.realX(x), surface.realY(y));
				    			
				    		if(this.ChosenInputNode!=null && node==null) 
				    		{
				    			NVertex C1 = ((NPanelNode)found).getCore().asCore();
				    			NVertex C2 = this.ConnectionPanelNeuron.getCore().asCore();
				    			C2.connect(C1, this.InputIndex);
				    		}
				    		this.ChosenInputNode = node;
				    			
				    		System.out.println("Chosen: "+this.ChosenInputNode);
				    		if(this.ChosenInputNode!=null && PU!=null) 
				    		{
				    			this.ConnectionPanelNeuron = PU;
				    			for(int i=0; i<PU.getInputNode().size(); i++) 
				    			{
				    				if(PU.getInputNode().get(i)==this.ChosenInputNode) {this.InputIndex = i;}
				    			}
				    		}
				    	}
				    }
				}
				surface.setClick(null);
				surface.repaint(0, 0, surface.getWidth(), surface.getHeight());
			}
		);
		Surface.setPressAction(
			(surface)->
			{
				int[] Press = surface.getPress();
				if(Press==null) {return;}
				int x = Press[0];
				int y = Press[1];
				System.out.println("Press: X:"+surface.realX((double)x)+"; Y:"+surface.realY((double)y)+";");						
				//surface.setPress(null);
			}
		);
		Surface.setLongPressAction(
			(surface)->
			{
				int[] LongPress = surface.getLongPress();
				if(LongPress==null || surface.getPress()!=null) 
				{
					surface.setPress(null);
					if(newMenu!=null) 
					{
						Surface.setMap(Surface.getMap().removeAndUpdate(newMenu));
						newMenu=null;
					}
					return;
				}
				int x = LongPress[0];
				int y = LongPress[1];
						
				//Object found = surface.findAnything(x, y, true, null);

				if(newMenu==null) //TODO: Presspoint sction on surface! 
				{
					newMenu = 
					new NCircleMenu(
					Surface.realX(x), 
				    Surface.realY(y), 
					Surface.getScale(), this);
						
					double[] frame = 
						{
							Surface.realX(0),Surface.realY(0), 
				            Surface.realX(Surface.getWidth()), Surface.realY(Surface.getHeight())
				        };
						    
					Surface.getMap().applyToAllWithin(frame, 
						(NPanelObject thing)->
						{
						    if(thing instanceof NCircleMenu) 
						    {
						    	if(thing.getLayerID()>=newMenu.getLayerID())
								{
						    		newMenu.setLayerID(thing.getLayerID()+1);
						    		System.out.println("mlid: "+newMenu.getLayerID());
						    		System.out.println("tlid: "+thing.getLayerID());
								}
						    }
						    return false;
						}
					);
					Surface.setMap(Surface.getMap().addAndUpdate(newMenu));
				}
				if(newMenu.animationState()>0) 
				{
					// PressPoint = null;
					Surface.setMap(Surface.getMap().addAndUpdate(newMenu));
					surface.getListener().PressPoint = null; 
					newMenu=null; 
				}
				surface.setLongPress(null);
			}
		);
		Surface.setDoubleClickAction(
			(surface)->
			{
				int[] DoubleClick = surface.getDoubleClick();
				if(DoubleClick==null) {return;}
					
				int x = DoubleClick[0];
				int y = DoubleClick[1];
				System.out.println("Double clicked at: "+x+", "+y);
					
				NPanelObject found = surface.findObject(surface.realX(x), surface.realY(y), true, null);
				if(found!=null) {found.doubleClickedAt(surface.realX(x), surface.realY(y), surface);}
				else
				{
					NPanelNode Unit = new NPanelNode(new NVertex_Root(this.BlueprintNode), surface.realX((double)x),surface.realY((double)y));
				 	if(surface.getMap()==null) 
				 	{
				 		surface.setMap(new NMap(Unit.getX(),Unit.getY(),10000));
				 	}
				 	surface.setMap(surface.getMap().addAndUpdate(Unit));
				 	Unit.setHasMoved(true);
				}
				surface.setDoubleClick(null);
			}
		);
		Surface.setScaleAction(
			(surface)->
			{
				if(surface.getScaling()==null) 
				{
					return;
				}
				int x = (int)surface.getScaling()[0];
				int y = (int)surface.getScaling()[1];
				double scaleFactor = surface.getScaling()[2];
					
				surface.setLastSenseX(x);
				surface.setLastSenseY(y);
					
				surface.translatePanel(-x, -y);
				surface.getScaler().scale((scaleFactor), (scaleFactor));
				surface.translatePanel(x, y);
				surface.setScaling(null);
				surface.repaint(0, 0, surface.getWidth(), surface.getHeight());
			}
		);
		Surface.setSenseAction(
			(surface)->
			{
				int[] Sense = surface.getSense();
				if(Sense==null) 
				{
					return;
				}
				int x = Sense[0];
				int y = Sense[1];
				surface.setMovement(null);
						
				double realX = surface.realX(x);
				double realY = surface.realY(y);
						
				if((this.MouseAttachedPanelNeuron!=null || this.MouseAttachedCircleMenu!=null) && surface.isInTouchMode() == false) 
				{
					//Listener.addDraggPoint(x, y);
				}
				if(surface.isInTouchMode()) 
				{
					this.MouseAttachedPanelNeuron=null; 
					this.MouseAttachedCircleMenu=null;
					//TODO: getLongPress and check distance...
					newMenu=null;
				}
				//This needs improvement! -> find neuron function needs to be optimized!
				if(surface.getFocusObject()!=null)
				{
					surface.getFocusObject().movementAt(realX, realY, surface);
				}
				//Overlapping nodes need to be considered!
				surface.setFocusObject(surface.findObject(realX, realY, true, null));
				if(surface.getFocusObject() instanceof NCircleMenu) 
				{

				}
				surface.setLastSenseX(x);
				surface.setLastSenseY(y);
			}		
		);
		Surface.setSwipeAction(
			(surface)->
			{
				int[] Swipe = surface.getSwipe();
				if(Swipe==null)
				{
					return;
				}	
				if(surface.isInTouchMode()) 
				{
					surface.setFocusObject(surface.findObject(surface.realX(Swipe[0]),surface.realY(Swipe[1]), true, null));
							
					if(surface.getFocusObject()!=null) 
					{
						ArrayList<NPanelRepaintSpace> RepaintQueue = surface.getRepaintQueue();
						if(RepaintQueue==null) {RepaintQueue = new ArrayList<NPanelRepaintSpace>();}
								
						surface.setLastSenseX(Swipe[2]);
						surface.setLastSenseY(Swipe[3]);

						double[] data = {surface.realX(Swipe[0]),surface.realY(Swipe[1]), surface.realX(Swipe[2]),surface.realY(Swipe[3])};
						RepaintQueue.addAll(surface.getFocusObject().moveDirectional(data, surface));
						surface.setRepaintQueue(RepaintQueue);
						Swipe[0]=Swipe[2];
						Swipe[1]=Swipe[3];
						Swipe=null; 
						return;
					}
					if(surface.getFocusObject()==null)
					{
						surface.translatePanel(Swipe[2]-Swipe[0],Swipe[3]-Swipe[1]);
						Swipe[0]=Swipe[2];
						Swipe[1]=Swipe[3];
						surface.setSwipe(null);
					}
				}
				else 
				{
					if(Swipe.length==4) 
					{
						surface.translatePanel(Swipe[2]-Swipe[0],Swipe[3]-Swipe[1]);
					}
					Swipe[0]=Swipe[2];
					Swipe[1]=Swipe[3];
					surface.setSwipe(null);
				}
				surface.setSwipe(null);
			}		
		);
		Surface.setDragAction(
			(surface)->
			{
				int[] Drag = surface.getDrag();
				if(Drag==null)
				{
					return;
				}
					
				if(surface.isInTouchMode()) 
				{
					int[] PressPoint = surface.getListener().PressPoint;
					if(PressPoint!=null) 
					{
						double distance = Math.pow(Math.pow(PressPoint[1]-Drag[0], 2)+Math.pow(PressPoint[2]-Drag[1], 2), 0.5);
						double mod = 1*(Math.pow(distance, (distance)));
						PressPoint[0] -= 1*(mod);
						if(PressPoint[0]<0) 
						{
							PressPoint[0]=0;
						}
						if(distance>pressRadius) 
						{
							if(newMenu!=null) 
							{
								surface.getListener().PressPoint = null; 
								Surface.setMap(Surface.getMap().removeAndUpdate(newMenu));
								newMenu=null;
							}
						}
						else 
						{
							if(newMenu!=null) {newMenu.modifyAnimationState((int)-mod);}
						}
						System.out.println("dragged at distance: "+distance+"; mod: "+mod);
					}	
				}
				else 
				{
							
				}
				surface.setSwipe(null);
			}
		);
	}
	
	
	//-------------------------------------------------------------------------
	 public int runNetwork(int currentStep) 
	 {
	 	if(Worker!=null) {if(Worker.isAlive()) {return currentStep;}}
		 setUpNewWorkThreader(5);
		  Worker = new Thread(ThreadCoordinator);
		  if(ThreadCoordinator==null) {return currentStep;}
		 if(Worker.isAlive()==false) {Worker.start();currentStep++;}  
		 return currentStep;
	 }
	 //-------------------------------------------------------------------------
	 public void randomizeEverything()
	 {
		 Surface.getMap().applyToAll(
			new neureka.ngui.NMapAPI.MapAction()
			{
				public boolean act(NPanelObject element) 
				{
					((NPanelNode)element).getCore().randomizeBiasAndWeight();;
					return false;
				}
			} 
		);
	 }
	//-------------------------------------------------------------------------
	 public void setUpNewWorkThreader(int HistoryScope) 
	 {	
	 	Object Data = new LinkedList<NVNode>();
	 	LinkedList<NVNode> applyAndGet =
			(LinkedList<NVNode>)Surface.getMap().applyAndGet(
			Data, (Object thing, NPanelObject other)
			 ->{
				   if(other instanceof NPanelNode) {((LinkedList<NVertex>)thing).add(((NPanelNode)other).getCore());}
				   return thing;
			   }
		  );
	 }
	//-------------------------------------------------------------------------
	 public void setHistoryScopeTo(int scope) 
	 {
	 	if(scope<0) {return;}
	 	Surface.getMap().applyToAll(
	 	(NPanelObject node)->
	 	{
	 		if(!(node instanceof neureka.ngui.NPanelNode)) {return false;}
	 		//NVCloak Properties = (NVCloak) ((NPanelNode)node).getCore().findModule(NVCloak.class);
			//if(Properties!=null) {
			//	NVMemory Memory = (NVMemory) ((NPanelNode)node).getCore().findModule(NVCloak.class);
			//	if(Memory!=null) {Memory.setTimeTunnel(scope);}}
		 return true;
		 });
	 }
	 //-------------------------------------------------------------------------
	
	
	
	
}
