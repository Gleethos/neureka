package ngui;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.MultipleGradientPaint.CycleMethod;
import java.awt.Paint;
import java.awt.Polygon;
import java.awt.RadialGradientPaint;
import java.awt.Rectangle;
import java.awt.geom.Area;
import java.awt.geom.Ellipse2D;
import java.awt.geom.GeneralPath;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.awt.geom.RoundRectangle2D;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;

import napi.main.core.NVCloak;
import napi.main.core.NVNode;
import napi.main.core.NVUtility;
import napi.main.core.NVertex;
import napi.main.core.imp.NVertex_Root;
import napi.main.core.mem.NVMemory;
import napi.main.core.opti.NVOptimizer;



public class NPanelNode implements NPanelObject
{

	private static int convectionTime = 500000000;
	
	private static final int   buttonLimit = 250000000;
	private static final int   deckAnimationLimit = 700000000;
	private static final float subToSuperScale = 5;
	
	private NVertex Core;
	private Color deckColor = 		new Color(0, 60, 110);//static!!!
	private Color darkDeckColor = 	new Color(0, 5,  60);//static!!
	private Color deckButtonColor = new Color(0, 0,  20, 255);//...
	private double[] position = new double[2];

	private static final int defaultDiameter = 1680;
	private int diameter =   defaultDiameter;
	
	private boolean deckIsRemoved = false;
	private boolean deckAnimationRunning = false;
	
	private boolean hoveringOverDeckButton = false;
	private boolean wasOnDeckButton = false;

	private boolean dataDisplayUpdateRequest = true;

	private boolean isOnDeck = false;
	private boolean wasOnDeck = false;
	private boolean hasRecentlyBeenMoved = false;
	private boolean hasJustBeenMoved = false;
	
	interface Painter 
	{public abstract void paint(Graphics2D painterBrush, double x, double y);}

	private ArrayList<NPanelNodeInput> InputNode;
	//===========================================================================
	NPanelNode(NVertex neuron, double x, double y)
	{
		Core = neuron;
		neuron.asCore().addModule(this);
		position[0] = x;
		position[1] = y;
		InputNode = new ArrayList<NPanelNodeInput>();
		InputNode.add(new NPanelNodeInput(getX(), getY(), diameter/2));
	}
	//===========================================================================
	//===========================================================================
	NPanelNode(double x, double y, int ID) 
	{
		position[0] = x;
		position[1] = y;
		InputNode = new ArrayList<NPanelNodeInput>();
		InputNode.add(new NPanelNodeInput(getX(), getY(), diameter/2));
		Core = new NVertex_Root(1, false, false, ID, 0, 15);
		Core.addModule(this);
		Core.addModule(new NVOptimizer(Core.size(), Core.getConnection(), 3));
		NVertex.State.turnIntoParentRoot(Core);
		//Core.turnIntoSuperRootNode();
	}
	//===========================================================================
	public void addInputNode() 
	{
		InputNode.add(new NPanelNodeInput(getX(), getY(), diameter/2));
		//InputField.get(InputField.size()-1).noticeChange();
		System.out.println("old connection input size: "+Core.inputSize());
		//Core.updateConnectionMatrix(Core.getInputSize(), false);
		Core.addInput(Core.inputSize());
		System.out.println("New connection input size: "+Core.inputSize());
		for(int Ii=0; Ii<InputNode.size(); Ii++) {InputNode.get(Ii).noticeChange();}
	}
	//===========================================================================
	public boolean justMoved() {return hasJustBeenMoved;}
	public boolean recentlyMoved() {return hasRecentlyBeenMoved;}
	public void setHasMoved(boolean moved) {hasJustBeenMoved = moved;}
    //------------------------------------------
	public NVertex getCore() {return Core;}
	public ArrayList<NPanelNodeInput> getInputNode() {return InputNode;}
	
	@Override
	public ArrayList<NPanelRepaintSpace> moveCircular(double[] data, NPanelAPI Surface) //double centerX, double centerY, double x, double y
	{
		if(data.length==3) {return moveCircularBy(data, Surface);}

		double centerX=data[0]; 
		double centerY=data[1]; 
		double x =     data[2]; 
		double y =     data[3];
		
		double vectorX = x-centerX;
		double vectorY = y-centerY;
		double distance = (((Math.pow(Math.pow(vectorX, 2)+Math.pow(vectorY, 2), 0.5))));
		vectorX /= distance;
		vectorY /= distance;
		//distance = (((Math.pow(Math.pow(position[0]-centerX, 2)+Math.pow(position[1]-centerY, 2), 0.5))));
		double oldX = position[0];
		double oldY = position[1];
		double newVecX = oldX-centerX;
		double newVecY = oldY-centerY;
		distance = (((Math.pow(Math.pow(newVecX, 2)+Math.pow(newVecY, 2), 0.5))));
		newVecX /= distance;
		newVecY /= distance;
		double alpha; 	//= Math.atan2(newVecX, newVecY)-Math.atan2(vectorX, vectorY);
		 alpha = -(Math.atan2(newVecY, newVecX)-Math.atan2(vectorY, vectorX));//Math.PI;
		 
		 double[] newData = {alpha, centerX, centerY}; 
		
		return moveCircularBy(newData, Surface);

	}

	
	private ArrayList<NPanelRepaintSpace> moveCircularBy(double[] data, NPanelAPI Surface) 
	{
		Surface.setMap(Surface.getMap().removeAndUpdate(this)); 
		 if(Surface.getMap()==null) 
		 {Surface.setMap(new NMap(getX(), getY(), 50000));}
		
		double alpha = data[0]; 
		double centerX = data[1]; 
		double centerY = data[2];
		ArrayList<NPanelRepaintSpace> queue = new ArrayList<NPanelRepaintSpace>();
			double vectorX = position[0]-centerX;
			double vectorY = position[1]-centerY;
			//double alpha = Math.atan2(newVecX, newVecY)-Math.atan2(vectorX, vectorY);
			for(int Ii=0; Ii<InputNode.size(); Ii++) 
		    {//InputNodevv.get(Ii).rotate(alpha);
			 queue.addAll(InputNode.get(Ii).adjustOrbit(getX(), getY(), diameter/2));
			 
		     queue.addAll(InputNode.get(Ii).moveCircular(data, Surface));//alpha, centerX, centerY
		    }
			queue.add(getRepaintSpace());
			position[0]=(centerX+(Math.cos(alpha)*(vectorX)-Math.sin(alpha)*(vectorY)));
			position[1]=(centerY+(Math.cos(alpha)*(vectorY)+Math.sin(alpha)*(vectorX)));
			queue.add(getRepaintSpace());
			
		if(Core.is(NVertex.MotherRoot)) 
		{//Moving circular around a given center!
			//calculating rotation!!! -> - => counter clock | + => clock wise
			NVNode[] nodes = Core.find((NVNode node)->{if(node.asCore().is(NVertex.Root) && node.asCore().is(NVertex.MotherRoot)==false) {return true;}return false;});
			if(nodes != null) 
			   {for(int Ni=0; Ni<nodes.length; Ni++) {for(int Nii=Ni+1; Nii<nodes.length; Nii++) {if(nodes[Ni]==nodes[Nii]) {nodes[Nii]=null;}}}
				for(int Ni=0; Ni<nodes.length; Ni++) {
					if(nodes[Ni]!=null) {
						NPanelNode panelNode = (NPanelNode) nodes[Ni].asCore().findModule(NPanelNode.class);
						if(panelNode!=null) {System.out.println("MoveCircular: ->sub root nodes move rotating: Ni "+Ni);
							vectorX = panelNode.getX()-position[2];
							vectorY = panelNode.getY()-position[3];
							data[1]=position[2];
							data[2]=position[3];
						    queue.addAll(panelNode.moveCircularBy(data, Surface));//alpha, position[2], position[3]
						   }
				       }
				   }
			   }
		}//--------------
		this.hasJustBeenMoved = true;

		 Surface.setMap(Surface.getMap().addAndUpdate(this)); 
		return queue;
	}
	@Override
	public ArrayList<NPanelRepaintSpace> moveDirectional(double[] data, NPanelAPI Surface) {//
		System.out.println("moveDirectional");
		//Surface.setObjectMap(Surface.getObjectMap().removeAndUpdate(this)); 
		// if(Surface.getObjectMap()==null) {Surface.setObjectMap(new NPanelMapBranch(getX(), getY(), 50000));}
		
	
		double startX=data[0];
		double startY=data[1]; 
		double targX=data[2]; 
		double targY=data[3];
		
		ArrayList<NPanelRepaintSpace> queue = new ArrayList<NPanelRepaintSpace>();
		hasJustBeenMoved = true;
		if(testForBody(startX, startY)) {
			double shiftX = startX-position[0];
			double shiftY = startY-position[1];
			
			if(Core.is(NVertex.MotherRoot)) {
				data[0]=position[2];
				data[1]=position[3];
				data[2]=targX;
				data[3]=targY;	
				
				queue.addAll(moveCircular(data, Surface));//Moving around the center!
			   }
			else 
			   {
				double[] newData = {(targX-shiftX),(targY-shiftY)};
				queue.addAll(moveTo(newData,Surface));
			   }
		}//else...
		else 
		{//test for nodes (set position only within bounds)
			if(Core.is(NVertex.MotherRoot)) {
				if(position.length==4) 
				{
					double distance =
					Math.pow((Math.pow((position[2]-startX), 2)+Math.pow((position[3]-startY), 2)), 0.5);
					//->Also: check if no other rim node has been touched! -> testFor!
					if(distance < 2.5*diameter) {
						queue.add(getRepaintSpace());
						double shiftX = targX-startX;
						double shiftY = targY-startY;
						NVNode[] RimNodes 
						= Core.find((NVNode node)->{if(node.asCore().is(NVertex.Root)&&node.asCore().is(NVertex.MotherRoot)==false){return true;}return false;});
						
						if(RimNodes != null) 
						   {for(int Ni=0; Ni<RimNodes.length; Ni++) {for(int Nii=Ni+1; Nii<RimNodes.length; Nii++) {if(RimNodes[Ni]==RimNodes[Nii]) {RimNodes[Nii]=null;}}}
							for(int Ni=0; Ni<RimNodes.length; Ni++) {System.out.println("Ni: "+Ni);
								if(RimNodes[Ni]!=null) {
									NPanelNode panelCompartment = (NPanelNode) RimNodes[Ni].asCore().findModule(NPanelNode.class);
									if(panelCompartment!=null) {

										 double[] newData = {(panelCompartment.getX()+shiftX),(panelCompartment.getY()+shiftY)};
									    queue.addAll(panelCompartment.moveTo(newData, Surface));
									   }
							       }
							   }
						   }
					queue.add(getRepaintSpace());
					Surface.setMap(Surface.getMap().removeAndUpdate(this)); 
					if(Surface.getMap()==null) {Surface.setMap(new NMap(getX(), getY(), 50000));}
					position[0]+=shiftX;
				    position[1]+=shiftY;
				    position[2]+=shiftX;
					position[3]+=shiftY;
					Surface.setMap(Surface.getMap().addAndUpdate(this)); 
					queue.add(getRepaintSpace());
					for(int Ii=0; Ii<InputNode.size(); Ii++) 
					{queue.addAll(InputNode.get(Ii).adjustOrbit(getX(), getY(), diameter/2));
					 double[] newData = {startX, startY, targX, targY};
					 queue.addAll(InputNode.get(Ii).moveDirectional(newData, Surface));}
					 
					   }
				}
			}
		}
		//
		return queue;
	}
	//---------------------
	@Override
	public ArrayList<NPanelRepaintSpace> moveTo(double[] data, NPanelAPI Surface) //
	{
		System.out.println("moveTo");
		Surface.setMap(Surface.getMap().removeAndUpdate(this)); 
		 if(Surface.getMap()==null) {Surface.setMap(new NMap(getX(), getY(), 50000));}
		 System.out.println("map updated");
		double x = data[0];
		double y = data[1];
		ArrayList<NPanelRepaintSpace> queue = new ArrayList<NPanelRepaintSpace>();
		
		if(Core.is(NVertex.MotherRoot)) {
			//SUB NEURON POSITION UPDATE!!!
			//moveCircular(position[2], position[3], x, y);
			queue.add(getRepaintSpace());
			double movedX = x-position[0];
		    double movedY = y-position[1];
			NVNode[] RimNodes = Core.find((NVNode node)->{if(node.asCore().is(NVertex.Root) && node.asCore().is(NVertex.MotherRoot)==false) {return true;}return false;});
			if(RimNodes != null) {
				for(int Ni=0; Ni<RimNodes.length; Ni++) {for(int Nii=Ni+1; Nii<RimNodes.length; Nii++) {if(RimNodes[Ni]==RimNodes[Nii]) {RimNodes[Ni]=null;}}}
				for(int Ni=0; Ni<RimNodes.length; Ni++) {
					if(RimNodes[Ni]!=null) {
						NPanelNode panelCompartment = (NPanelNode) RimNodes[Ni].asCore().findModule(NPanelNode.class);
						if(panelCompartment!=null) {
							data[0]=panelCompartment.getX()+movedX;
							data[1]=panelCompartment.getY()+movedY;
						    queue.addAll(panelCompartment.moveTo(data, Surface));
						}
				   }
				}
			}
		    if(position.length==4) 
		       {
			      position[0] = x;
			      position[1] = y;
			      position[2]+=movedX;
				  position[3]+=movedY;
		       }
		    for(int Ii=0; Ii<InputNode.size(); Ii++) 
				{queue.addAll(InputNode.get(Ii).adjustOrbit(getX(), getY(), diameter/2));
				data = new double[4];
				data[0]=position[0];
				data[1]=position[1];
				data[2]=x;
				data[3]=y;
		    	queue.addAll(InputNode.get(Ii).moveDirectional(data, Surface));}
		}
		else 
		{
		queue.add(getRepaintSpace());
		for(int Ii=0; Ii<InputNode.size(); Ii++) 
		{queue.addAll(InputNode.get(Ii).adjustOrbit(getX(), getY(), diameter/2));
		data = new double[4];
		data[0]=position[0];
		data[1]=position[1];
		data[2]=x;
		data[3]=y;
    	queue.addAll(InputNode.get(Ii).moveDirectional(data, Surface));}
		
		position[0] = x;
		position[1] = y;
		
		queue.add(getRepaintSpace());
		}
		hasJustBeenMoved = true;
		System.out.println("hey cool");
		Surface.setMap(Surface.getMap().addAndUpdate(this)); 
		System.out.println("niiiiccceeee");
		return queue;
	}
	// --------------------------------------------------------------------------------------------
	public NPanelRepaintSpace getRepaintSpace() 
	{
		NPanelRepaintSpace repaintSpace = null;
		if(Core.is(NVertex.MotherRoot)) 
		   {repaintSpace = new NPanelRepaintSpace(position[2], position[3], (diameter/2)*subToSuperScale*1.2, (diameter/2)*subToSuperScale*1.2);}
		else                         
		   {double r = (diameter/2)*(1.1+(1-(diameter/defaultDiameter))/1.995);
			repaintSpace = new NPanelRepaintSpace(position[0], position[1], r, r);}
		return repaintSpace;
	}
	// --------------------------------------------------------------------------------------------
	public void setInputActivity(boolean[] activity) 
	{
		if(activity.length==1) 
		   {
				for(int Ii=0; Ii<InputNode.size(); Ii++) 
			    	{InputNode.get(Ii).setIsActive(activity[0]);}
		   }
		else
		   {	
			    for(int Ii=0; Ii<InputNode.size(); Ii++)
		            {InputNode.get(Ii).setIsActive(activity[Ii]);}
		   }
		System.out.println(activity.length+"setting input activity...");
	}
	// --------------------------------------------------------------------------------------------
	//public ArrayList<NPanelRepaintSpace> update
	
	public ArrayList<NPanelRepaintSpace> updateOn(NPanelAPI HostPanel) 
	{
		//System.out.println("wasOnDeck:"+wasOnDeck+"; isOnDeck:"+isOnDeck+";");
		boolean scaleCheck = true;
		if(HostPanel.getScale()<0.2) {scaleCheck=false;}//
		if(HostPanel.getScale()<0.13 && deckIsRemoved) {deckAnimationRunning=true;}
		
		ArrayList<NPanelRepaintSpace> queue = new ArrayList<NPanelRepaintSpace>();
		 for(int Ii=0; Ii<InputNode.size(); Ii++) {queue.addAll(InputNode.get(Ii).adjustOrbit(getX(), getY(), diameter/2));}
		 
		if(Core.is(NVertex.MotherRoot)) 
		{
			if(position.length==2) {
				double[] old = position;
				position = new double[4];
				position[0]=old[0];
				position[1]=old[1];
				position[2]=old[0];
				position[3]=old[1];
			}
			NVNode[] RimNodes = NVUtility.identityCorrected.order(Core.find(
					(NVNode node)->{if(node.asCore().is(NVertex.Root)||node.asCore()==this.Core) {return true;}return false;}));
			
			if(RimNodes != null) 
			   {//System.out.println("Rim nodes exist! -> "+RimNodes.length);
				 boolean updateNeeded = false;
				   for(int Ni=0; Ni<RimNodes.length; Ni++) 
				       {
					    	if(RimNodes[Ni]!=null) 
					        {
					           NPanelNode panelNode = (NPanelNode) RimNodes[Ni].asCore().findModule(NPanelNode.class);
					           if(panelNode!=null) 
					           {//optimization: Checking if nodes have been moved...
					              if(panelNode.justMoved()) {updateNeeded=true;}//Needed?!?
					       
					              if(RimNodes[Ni].asCore().is(NVertex.RootInput) || RimNodes[Ni].asCore().is(NVertex.MotherRoot)) 
					              {		
					        	   		double directionX = panelNode.getX()-position[2]; 
						        	   	double directionY = panelNode.getY()-position[3];
					        	   		double distance = Math.pow(Math.pow(directionX,2)+Math.pow(directionY, 2), 0.5);
				
					        	   		if(distance > diameter*2.5 || distance < diameter*2.25 || updateNeeded) 
					        	   		{
					        	   			directionX/=distance;
					        	   			directionY/=distance;
					        	   			if(distance == 0) 
					        	   			{
					        	   				Random dice = new Random();
					        	   				boolean sign = dice.nextBoolean();
					        	   				directionX=dice.nextDouble()%diameter*0.1;if(sign) {directionX*=-1;}
					        	   				sign = dice.nextBoolean();
					        	   				directionY=dice.nextDouble()%diameter*0.1;if(sign) {directionY*=-1;}
					        	   			}
					        	   			double delta=0;
					        	   			if(distance > diameter*2.5) {delta=-(distance-diameter*2.25)*0.5;}else {delta=(diameter*2.26-distance)*0.5;}
					        	   			
					        	   				for(int Ii=0; Ii<InputNode.size(); Ii++) 
					        	   					{	
					        	   						double a = Math.abs(InputNode.get(Ii).angleOf(getX(), getY())-InputNode.get(Ii).angleOf(getX(), getY()));
					        	   						if(a > Math.PI/6) 
					        	   						{
					        	   							//queue.add(InputNode.get(Ii).getRepaintSpace(getX(), getY(), diameter/2));
					      	   							//InputNode.get(Ii).rotateTowards(-directionX, -directionY);
					        	   						//queue.addAll(InputNode.get(Ii).adjustWithRespectTo(-directionX, -directionY, getX(), getY(), InputNode));
					        	   						double[] newData = {InputNode.get(Ii).getX(), InputNode.get(Ii).getY(), position[2], position[3]}; 
					        	   						
					        	   						queue.addAll(InputNode.get(Ii).moveCircular(newData, HostPanel));
					        	   							
					        	   						
					        	   					   }
					        	   						else {
					        	   							queue.addAll(InputNode.get(Ii).adjustOrbit(getX(), getY(), diameter/2));
					        	   						}
					        	   					}
					        	   				
					        	   				if(panelNode == this) {
					        	   						//System.out.println("Adjustment! for rim node...");
					        	   						position[0]+=directionX*delta;
					        	   						position[1]+=directionY*delta;
					        	   						//queue.add(new NPanelRepaintSpace((position[0]), (position[1]), (1.3*diameter/2), (1.3*diameter/2)));
					        	   				}
					        	   				else
					        	   				{
					        	   				 queue.add(panelNode.getRepaintSpace());	
					        	   				 double[] data = {panelNode.getX()+directionX*delta, panelNode.getY()+directionY*delta};
	
					        	   				 queue.addAll(panelNode.moveTo(data, HostPanel));//queue.addAll<- This might not be needed!
					        	   				}
					        	   				
					        	   		   }//=>IMPORTANT: Repaint queue must be filled accordingly!
					              	}
					           	}
					        }
					    }
			}
		//Getting input node rotations!
		//->adding them together!
		//drawing a core body accordingly!
		//->sizing input nodes according to core connections...
		
		//choosing coloring.... Suggestion: ?
		}
		else
		{
			if(position.length!=2) {
				double[] old = position;
				position = new double[2];
				position[0]=old[0];
				position[1]=old[1];
			}
		}
		//if(animationsExist){}
		queue.addAll(updateAnimations(HostPanel));
		
		
		if (dataDisplayUpdateRequest) {
			dataDisplayUpdateRequest = false;
			queue.add(new NPanelRepaintSpace((position[0]), (position[1]), (420), (310)));
		}

		if     (hasJustBeenMoved==true && hasRecentlyBeenMoved==false) {hasRecentlyBeenMoved = true;}
		else if(hasJustBeenMoved==true && hasRecentlyBeenMoved==true)  {hasJustBeenMoved     = false;}
		else if(hasJustBeenMoved==false && hasRecentlyBeenMoved==true) {hasRecentlyBeenMoved = false;}
		//System.out.println("All done ");
		return queue;
	}
	//--------------------------
	
	private ArrayList<NPanelRepaintSpace> updateAnimations(NPanelAPI HostPanel) {

			NAnimator Animator = HostPanel.getAnimator();
			int frameDelta = HostPanel.getCurrentFrameDelta();
			boolean scaleCheck = true;
			if(HostPanel.getScale()<0.2) {scaleCheck=false;}//
			if(HostPanel.getScale()<0.13 && deckIsRemoved) {deckAnimationRunning=true;}
			
			boolean UnitNeedsToBeRepainted = false;
			
			ArrayList<NPanelRepaintSpace> queue = new ArrayList<NPanelRepaintSpace>();
			
			int animationID = 0;
			// boolean deckCheck=false;
			// Animation stack! -> counter counts only animated cases.
			//=================================================================================================================================================
			//Animation 0 -> Deck Button
			//=================================================================================================================================================
			if (wasOnDeck) 
			{animationID = 0;
				//System.out.println("was on deck: "+Animator.getCounterOf(this, animationIDCounter));
				if(Animator.hasCounter(this, animationID)) 
				   {
				   	    if (Animator.getCounterOf(this, animationID) < buttonLimit && Animator.getCounterOf(this, animationID) > 0) {
						double mod = ((double)Animator.getCounterOf(this, 0)/buttonLimit);
						queue.add(new NPanelRepaintSpace((position[0]), (position[1]-0.0625*diameter*mod), (diameter/4), (diameter/8)+0.065*diameter*mod));
						}
				   }
				
				if (Animator.hasCounter(this, animationID)) 
				   {//System.out.println("Deck counter existing");
					if (isOnDeck && scaleCheck) 
					{
						//System.out.println("update->wasondeck->Is on deck");
						Animator.countUpFor(this, frameDelta,  animationID);
						if (Animator.getCounterOf(this, animationID) > buttonLimit) {//100
							Animator.countDownFor(this, frameDelta, animationID);
						}
					}
					else 
					{
						if (Animator.getCounterOf(this, animationID) >= 0) {
							Animator.countDownFor(this, frameDelta, animationID);
						}
						//System.out.println("wasOnDeckButton:"+wasOnDeckButton);
						if (Animator.getCounterOf(this, animationID) <= 0 && deckAnimationRunning==false && wasOnDeckButton == false) {//
							Animator.removeCounterOf(this, animationID);
							//System.out.println("removed deck counter...");
							wasOnDeck = false;
							//animationID--;
						}
					}
				} 
				else 
				{
					if (isOnDeck) {
						Animator.setCounterFor(this, animationID, 1); // System.out.println("Deck counter set!");
						wasOnDeck=true;
					}

				}
				if(deckAnimationRunning) {
					if(Animator.hasCounter(this, animationID)) {Animator.countDownFor(this, 6*frameDelta,0);}   
				}
				if(wasOnDeck==true && Animator.getCounterOf(this, animationID)<=0) 
				   {Animator.countUpFor(this, 1-Animator.getCounterOf(this, animationID),animationID);}
				
				//if(wasOnDeck==true && Animator.getCounterOf(this, animationID)>=0) {animationID++;}
			}
			//=================================================================================================================================================
			// Animation 1 -> Deck button sense
			//=================================================================================================================================================
			if (wasOnDeckButton && wasOnDeck) {//
				animationID=1;
				//System.out.println("Button Shining: "+Animator.getCounterOf(this, animationIDCounter));
				if (Animator.hasCounter(this, animationID)) {
					
					if (hoveringOverDeckButton && scaleCheck) {
						//System.out.println("is hovering on deck button"+Animator.getCounterOf(this, animationIDCounter));
							Animator.countUpFor(this, (frameDelta/6), animationID);
							if (Animator.getCounterOf(this, animationID) >= 2*buttonLimit) {//255
								Animator.countDownFor(this, (frameDelta/6), animationID);
								}
						//System.out.println("Button sense counter: "+Animator.getCounterOf(this, animationIDCounter)+"..."+animationIDCounter);
							double mod = ((double)Animator.getCounterOf(this, animationID)/(double)(2*buttonLimit));
							if(mod>1) {mod=1;}//System.out.println("Mod"+mod);
							deckButtonColor = new Color(0, 255, 255, (int) (255*mod));
							//if (Animator.getCounterOf(this, animationIDCounter) > 0) {queue.add(new NPanelRepaintSpace((position[0]), (position[1]), (110), (55)));}
					} 
					else
					{//System.out.println("is off deck button");
							if (Animator.getCounterOf(this, animationID) >= 0) {
								double mod = ((double)Animator.getCounterOf(this, animationID)/(double)(2*buttonLimit));
								if(mod>1) {mod=1;}else if(mod<0) {mod=0;}
								deckButtonColor = new Color(0, 255, 255, (int) (255*mod));
								Animator.countDownFor(this, (frameDelta/4), animationID);
							}
							
						if (Animator.getCounterOf(this, animationID) <= 0 && deckAnimationRunning==false) {
							Animator.removeCounterOf(this, animationID);
							deckButtonColor = new Color(0, 255, 255, (0));
							wasOnDeckButton = false;
							//animationID--;
						}
					}
					
				} 
				else 
				{
					if (hoveringOverDeckButton) {
						Animator.setCounterFor(this, animationID, 1);
						// System.out.println("Deck button counter set!");
					}
					
				}
				if(deckAnimationRunning) 
				{if(Animator.hasCounter(this, animationID)) {Animator.countDownFor(this, 6*frameDelta,animationID);
			
				if(Animator.getCounterOf(this, animationID)<=0) {Animator.countUpFor(this, 1-Animator.getCounterOf(this, animationID),animationID);}
			
				//if(Animator.getCounterOf(this, animationID)>=0) {animationID++;}
				}}
				
			}
			//=================================================================================================================================================
			//animation 2 -> deck opening
			//=================================================================================================================================================
			if(deckAnimationRunning) 
			{
				//System.out.println("deck Animation: "+Animator.getCounterOf(this, animationIDCounter));
				animationID = 2;
				if (Animator.hasCounter(this, animationID)) {
					Animator.countUpFor(this, frameDelta, animationID);

					//System.out.println("deckCou: "+Animator.getCounterOf(this, animationIDCounter));
					UnitNeedsToBeRepainted=true;
					
					if(Animator.getCounterOf(this, animationID)>deckAnimationLimit) //5000000
					   {Animator.removeCounterOf(this, animationID);
					    if(deckIsRemoved) {deckIsRemoved=false;}else{deckIsRemoved=true;}
					   // animationID--; 
					    deckAnimationRunning = false;
					   }
					
					}
				else 
					{Animator.setCounterFor(this, animationID, 1);
					 //Somehow the animation ID is 2 sometimes although so many animations are not even set!
					}
				//if(Animator.getCounterOf(this, animationID)>=0) {animationID++;}
			}
			//=================================================================================================================================================
			//Animation 3 -> Weight convection
			//=================================================================================================================================================
			boolean activityCheck = false;
			for(int Ii=0; Ii<InputNode.size(); Ii++) 
			    {
				 	if(InputNode.get(Ii).isActive()) {activityCheck=true; 
				 	//System.out.println("Input activity found... => convection");
				 	Ii=InputNode.size();}
				}
			ArrayList<NPanelRepaintSpace> newQueue = updateConnectionVectorAndGetRepaintSpace(activityCheck);
			if(newQueue!=null) {for(int Ri=0; Ri<newQueue.size(); Ri++) {queue.add(newQueue.get(Ri));}}
			if(activityCheck) 
			{
				animationID = 3;
				//Error was here => IDCounter was -1!!! Note: This has something to do with the deck animation!
				// Animation 1 -> Deck button sense
				if (Animator.getCounterOf(this, animationID) >= 0) 
				{
					Animator.countUpFor(this, frameDelta, animationID);
					//System.out.println("conCou: "+Animator.getCounterOf(this, animationIDCounter));
					if(Animator.getCounterOf(this, animationID)>convectionTime) //5000000
					{
						Animator.removeCounterOf(this, animationID);
						for(int Ii=0; Ii<InputNode.size(); Ii++) {InputNode.get(Ii).setIsActive(false);}
						    //animationID--;
				    }
				}
				else 
				{Animator.setCounterFor(this, animationID, 1);}
				//animationID++;
			}
			
			if(this.hasJustBeenMoved || UnitNeedsToBeRepainted) 
			   {queue.add(this.getRepaintSpace());
			    for(int Ii=0; Ii<InputNode.size(); Ii++) {queue.add(InputNode.get(Ii).getRepaintSpace(diameter/2));}
			   }
			else 
			   {for(int Ii=0; Ii<InputNode.size(); Ii++){if(InputNode.get(Ii).changeOccured()) {queue.add(InputNode.get(Ii).getRepaintSpace(diameter/2));}}}
			
			return queue;
	}
	// --------------------------------------------------------------------------------------------
	//public synchronized void requestWeightConvectionAnimation() {
		//weightConvectionRequest = true;
	//}
	// --------------------------------------------------------------------------------------------
	public synchronized void requestDataDisplayUpdate()
	{
		dataDisplayUpdateRequest = true;
	}
	// --------------------------------------------------------------------------------------------
	// ============================================================================================
	// INTERACTION:
	// --------------------------------------------------------------------------------------------
	public void movementAt(double trueMouseX, double trueMouseY, NPanelAPI HostSurface) 
	{// System.out.println("Is on deck:
		double panelScale = HostSurface.getScale();																// "+isOnDeck);
		if (testForBody(trueMouseX, trueMouseY)) 
		{isOnDeck  = true;wasOnDeck = true;}
		else 
		{isOnDeck = false;}

		if (testForButton_Open(trueMouseX, trueMouseY, panelScale)) 
		{hoveringOverDeckButton = true;}

		for (int i = 0; i < InputNode.size(); i++) {
			if (InputNode.get(i).testFor(trueMouseX, trueMouseY, diameter / 2)) 
			    {
				    if (isOnDeck == true) {wasOnDeck = true;}
				    isOnDeck = true;
				}
		}
		//was on deck = false ->somewhere her maybe?
	}

	// --------------------------------------------------------------------------------------------
	@Override
	public boolean clickedAt(double x, double y, NPanelAPI HostPanel) 
	{
		System.out.println("clicked!");
		if(deckAnimationRunning==true) 
		{
			return false;
		}
		if(testForButton_Open((x), (y), HostPanel.getScale())) 
        {//this.getNeuron().getPropertyHub().addProperty(new NNeuronControlFrame("TEST", this.getNeuron()));
			this.deckAnimationRunning = true;
        }
		return true;
     
	}
	@Override
	public boolean doubleClickedAt(double x, double y, NPanelAPI HostPanel) 
	{
		System.out.println("double clicked!!");
		boolean check = false;
		for (int i = 0; i < InputNode.size(); i++) 
		{   
			System.out.println("lol");
			if (InputNode.get(i).testFor(x, y, (diameter / 2))) 
			{
				check = true;
			}
		}
		if(check==true) 
		{
			System.out.println("found!!!");
			Core.addInput(Core.inputSize());
			this.InputNode.add(new NPanelNodeInput(getX(), getY(), diameter/2));
		}
		return false;
	}
	// SENSING:
	// ============================================================================================
	// --------------------------------------------------------------------------------------------
	public boolean isAttachable(double trueX, double trueY, double panelScale) {
		
		if(deckAnimationRunning==false) 
		{
			if (testForButton_Open(trueX, trueY, panelScale)) 
			{
				return false;
			}
		}

		for (int i = 0; i < InputNode.size(); i++) 
		{
			if (InputNode.get(i).testFor(trueX, trueY, diameter / 2)) 
			{
				return false;
			}
		}
		//if super root -> check rim nodes!
		if(Core.is(NVertex.MotherRoot)) 
		{
			double d = Math.pow(Math.pow((trueX - position[2]), 2) + Math.pow((trueY - position[3]), 2), 0.5);
			if(d <= diameter*2.5) 
			{
				return true;
			}
		}
		return true;
	}
	// TESTFOR FUNCTIONS:
	// --------------------------------------------------------------------------------------------
	public boolean testForBody(double x, double y) 
	{
		//NODE IS SUPER ROOT NODE! -> conditions!
		double d = Math.pow(Math.pow((x - position[0]), 2) + Math.pow((y - position[1]), 2), 0.5);
		for (int i = 0; i < InputNode.size(); i++) {
			 if (InputNode.get(i).testFor(x, y, (diameter / 2))) 
			     {
				    return true;
			     }
		}
		if (d <= diameter / 2) {return true;}
		return false;
	}
	// --------------------------------------------------------------------------------------------
	public boolean testForBody_Root(double x, double y) 
	{
		if(Core.is(NVertex.MotherRoot) && position.length==4) 
		{
			double d = Math.pow(Math.pow((x - position[2]), 2) + Math.pow((y - position[3]), 2), 0.5);
			if(d <= diameter*2.5) 
			{
				return true;
			}
		}
		return false;
	}

	// --------------------------------------------------------------------------------------------
	public boolean testForButton_Open(double trueMouseX, double trueMouseY, double panelScale) 
	{
		if(this.deckAnimationRunning) 
		{
			hoveringOverDeckButton=false;
			isOnDeck=false;
			return false;
		}
		double modifier = panelScaleModifier(panelScale);
		int x = (int) position[0];
		int y = (int) position[1];
		boolean check = true;
		if (trueMouseX < x - (int) (diameter / 4 * Math.pow(modifier, 100 / modifier)) / 2) {
			check = false;
		}
		if (trueMouseY < y - (int) (diameter / 8 * Math.pow(modifier, 100 / modifier)) / 2) {
			check = false;
		}
		if (trueMouseX > x + (int) (diameter / 4 * Math.pow(modifier, 100 / modifier)) / 2) {
			check = false;
		}
		if (trueMouseY > y + (int) (diameter / 8 * Math.pow(modifier, 100 / modifier)) / 2) {
			check = false;
		}
		if (hoveringOverDeckButton == true && deckAnimationRunning==false) 
		{
			wasOnDeckButton = hoveringOverDeckButton;
		}
		hoveringOverDeckButton = check;
		if (check && deckAnimationRunning==false) 
		{
			wasOnDeckButton = true;
		}
		return check;
	}
	// --------------------------------------------------------------------------------------------
	public NPanelNodeInput testFor_AndGet_InputNode(double trueX, double trueY) 
	{
		for (int i = 0; i < InputNode.size(); i++) 
		{
			if (InputNode.get(i).testFor(trueX, trueY, diameter / 2)) 
			{
				return InputNode.get(i);
			}
		}
		return null;
	}

	//CONNECTION REPAINT SPACE
	//=============================================================================================
	// --------------------------------------------------------------------------------------------
	private ArrayList<NPanelRepaintSpace> updateConnectionVectorAndGetRepaintSpace(boolean convection) // OPTIMIZATION?
	{//System.out.println("convection: "+convection);
		//-> checking if connection points are within frame!
		double vX = 0;
		double vY = 0;
		ArrayList<NPanelRepaintSpace> queue = new ArrayList<NPanelRepaintSpace>();;
		NVNode[][] connection = Core.getConnection();
		if (connection != null) 
		{
			//System.out.println("I size:"+connection.length);
			for (int Ii = 0; Ii < connection.length; Ii++) 
			{
				//System.out.println("Ii:"+Ii);
				boolean connectionMoved=this.justMoved();
				if (connection[Ii] != null && connectionMoved==false) 
				{
					for (int Ni = 0; Ni < connection[Ii].length; Ni++) 
					{
						if(connection[Ii][Ni]!=null) {
							//if(this.getNeuron().isBasicRootNode()) {System.out.println("Bananas baby");}
						   if(((NPanelNode)connection[Ii][Ni].asCore().findModule(NPanelNode.class)).justMoved()) {
							   connectionMoved=true;
								Ni=connection[Ii].length;
						   		}
					        }
					}
				}
					if(connectionMoved || InputNode.get(Ii).changeOccured() || convection) 
					{//=================================================================
						//System.out.println("convection: "+convection);
					 if(connection[Ii]!=null) {	
						for (int Ni = 0; Ni < connection[Ii].length; Ni++) 
						{
							if(connection[Ii][Ni]!=null) 
						  	{
							    vX += ((NPanelNode)connection[Ii][Ni].asCore().findModule(NPanelNode.class)).getX() - getX();
						        vY += ((NPanelNode)connection[Ii][Ni].asCore().findModule(NPanelNode.class)).getY() - getY();
						    }
						  //...
						}
					if (vX == 0 && vY == 0) 
					{
						Random dice = new Random();
						vX=dice.nextDouble()%1;vY=dice.nextDouble()%1;
					} 
					else 
					{
						double alpha = Math.PI+(InputNode.get(Ii).angleOf(getX(), getY())-Math.atan2(vY, -vX));//!!!!!!!!!!!!!!!!!!!!!!!
						//alpha = -(Math.atan2(vY, vX)-Math.atan2(InputNode.get(Ii), vectorX));
						double[] newData = {alpha, getX(), getY()};
						//queue.addAll(InputNode.get(Ii).adjustOrbit(getX(), getY(), diameter/2));
						queue.addAll(InputNode.get(Ii).moveCircular(newData, null));
						
						//for(int i=0; i<InputField.size(); i++) {InputField.get(i).adjustWithRespectTo(vX, vY, InputField);}
						if (InputNode.get(Ii).changeOccured() || convection) {
						    for (int Ni = 0; Ni < connection[Ii].length; Ni++) {
								 
								 if (connection[Ii][Ni]!=null)
								    {
									 double connectionX = ((NPanelNode)connection[Ii][Ni].asCore().findModule(NPanelNode.class)).getX();
									 double connectionY = ((NPanelNode)connection[Ii][Ni].asCore().findModule(NPanelNode.class)).getY();
									 if(dataDisplayUpdateRequest) {
										 double midPX = InputNode.get(Ii).getX() + (connectionX - InputNode.get(Ii).getX())/3.25;
										 double midPY = InputNode.get(Ii).getY() + (connectionY - InputNode.get(Ii).getY())/3.25;
										 //if (connectionSpace == null) {connectionSpace = new ArrayList<NPanelRepaintSpace>();}	
										 //WEIGHT REPAINT:
										 queue.add(new NPanelRepaintSpace((midPX),(midPY), 50, 25));
									 	}
									 double vectorX = (connectionX - getX());
									 double vectorY = (connectionY - getY());
									 double repaintCenterX = getX() + (vectorX) / 2;
									 double repaintCenterY = getY() + (vectorY) / 2;
									 double distanceX = Math.abs((connectionX - getX()) / 2);
									 double distanceY = Math.abs((connectionY - getY()) / 2);
									 if(convection) {
										 double d = Math.pow(Math.pow(vectorX, 2)+Math.pow(vectorY, 2), 0.5);
										 double v0X = vectorX/d;
										 double v0Y = vectorY/d;
										 
										 double normalX = (-v0Y)*diameter/2;
										 double normalY = (v0X)*diameter/2;
											double sideVecOneX = (connectionX+normalX)-repaintCenterX; 
											double sideVecOneY = (connectionY+normalY)-repaintCenterY;
											
											double sideVecTwoX = (connectionX-normalX)-repaintCenterX; 
											double sideVecTwoY = (connectionY-normalY)-repaintCenterY;
											if(sideVecOneX>distanceX) {distanceX=sideVecOneX;}
											if(sideVecTwoX>distanceX) {distanceX=sideVecTwoX;}
											if(sideVecOneY>distanceY) {distanceY=sideVecOneY;}
											if(sideVecTwoY>distanceY) {distanceY=sideVecTwoY;}
									 }//System.out.println("added repaint ...");
								     queue.add(new NPanelRepaintSpace(
										repaintCenterX,
										repaintCenterY,
										distanceX,
										distanceY));	
									}
								}	
						      if(convection==false) 
						      {
						    		//System.out.println("Gottcha!");
						    		queue.add(InputNode.get(Ii).getRepaintSpace(diameter/2));	
						    		//InputNode.get(Ii).forgetChange();	
						       }
						       InputNode.get(Ii).forgetChange();	
						   }   
					   }
					   vX = 0;
					   vY = 0;
					}//else{System.out.println("Connection is null wtf!!!!");}
				}//=================================================================
			}
		}
		return queue;
	}
	//PAINTING:
	//--------------------------------------------------------------------------------------------
	//=============================================================================================
	private void paintNodeConnections(Graphics2D brush,  NAnimator Animator) 
	{
		Color connectionColor = Color.CYAN;
		int animationID   = 0;
		int convectionCounter    = 0;

        boolean activityCheck = false;
		for(int Ii=0; Ii<InputNode.size(); Ii++) 
		{
			if(InputNode.get(Ii).isActive()) 
			{
				activityCheck=true; 
				Ii=InputNode.size();
			}
		}
		animationID   = 3;
		if(activityCheck) 
		{
			convectionCounter = Animator.getCounterOf(this, animationID);
		}
		//=======================================================================================================================
		NVNode[][] connection = Core.getConnection();
		if (connection != null) 
		{
			for (int Ii = 0; Ii < connection.length; Ii++) 
			{
				if (connection[Ii] != null) 
				{
					double selfCenterX = InputNode.get(Ii).getX();
					double selfCenterY = InputNode.get(Ii).getY();
					double inputVecX = selfCenterX - getX();
					double inputVecY = selfCenterY - getY();
					
					if(Core.is(NVertex.Root)) 
					{
						connectionColor=NColor.SystemOcean;
					    if(Core.is(NVertex.RootInput)) 
					       {if(!connection[Ii][0].asCore().is(NVertex.Root) || connection[Ii][0].asCore().is(NVertex.MotherRoot)) 
					           {connectionColor = Color.CYAN;}
					       }
					}
					for (int Ni = 0; Ni < connection[Ii].length; Ni++) 
					{
						if(connection[Ii][Ni]!=null) 
						{
							double connectionX = ((NPanelNode)connection[Ii][Ni].asCore().findModule(NPanelNode.class)).getX();
							double connectionY = ((NPanelNode)connection[Ii][Ni].asCore().findModule(NPanelNode.class)).getY();
							
							GeneralPath bezier = new GeneralPath();
							
							double vectorX = (connectionX - selfCenterX);
							double vectorY = (connectionY - selfCenterY);
							
							double otherX = selfCenterX+vectorX;
							double otherY = selfCenterY+vectorY;
							
							NUtility math = new NUtility();
							
							double deg = math.unitaryVectorProduct(vectorX, vectorY, inputVecX, inputVecY); 
							//System.out.println(deg);
							
							
							double outerX = selfCenterX+inputVecX*(1+Math.abs(deg));
							double outerY = selfCenterY+inputVecY*(1+Math.abs(deg));
							deg = 1.5*(1-deg);
							
							double orthoX, orthoY;
							if(math.unitaryVectorProduct(vectorX, vectorY, -inputVecY, inputVecX)<0) 
							{
								orthoX = outerX+deg * inputVecY;
								orthoY = outerY-deg * inputVecX;
							}
							else 
							{
								orthoX = outerX-deg * inputVecY;
								orthoY = outerY+deg * inputVecX;
								
							}
							double[] points = {selfCenterX, selfCenterY, 
						     		           outerX, outerY, 
						     		           orthoX, orthoY,
						                       otherX, otherY};
							
							bezier.moveTo( points[0], points[1]);
							bezier.curveTo(points[2], points[3], 
									       points[4], points[5],
									       points[6], points[7]);
							
							
							brush.setColor(connectionColor);
							brush.setStroke(new BasicStroke(22));
							brush.draw(bezier);
							
							
					
							double[] curvePoint = null;
							double ratio = 1-((double)(convectionCounter))/convectionTime;
							
							if(activityCheck) 
							{
								NPanelNode Other = ((NPanelNode)connection[Ii][Ni].asCore().findModule(NPanelNode.class));
								
								double d = Math.pow(Math.pow(vectorX, 2)+Math.pow(vectorY, 2), 0.5);
								//System.out.println(diameter);
								double normalX = (-vectorY/d)*Other.diameter/2*ratio;
								double normalY = (vectorX/d)*Other.diameter/2*ratio;
								Polygon convector = new Polygon();//System.out.println("painting convection!"+ratio);
								double invratio = 1-ratio;
								curvePoint = math.getCurvePointOn(invratio*invratio*invratio, points);
								convector.addPoint((int)(curvePoint[0]), 
										           (int)(curvePoint[1]));
								curvePoint = math.getCurvePointOn(invratio, points);//=>Convector center
								convector.addPoint((int)(curvePoint[0]+normalX), (int)(curvePoint[1]+normalY));
								convector.addPoint((int)(selfCenterX), (int)(selfCenterY));
								convector.addPoint((int)(curvePoint[0]-normalX), (int)(curvePoint[1]-normalY));
			
								brush.fillPolygon(convector);
							}
							curvePoint = math.getCurvePointOn(1/3.25, points);
							if(this.Core.hasWeight(Ii)) 
							{
								String value = ""+ getCore().getWeight(Ii, Ni);
								brush.fillOval((int)(curvePoint[0])-70, (int)(curvePoint[1])-25, 140, 50);
								brush.setColor(Color.BLACK);
						
								brush.drawString(value,(int) (curvePoint[0])-value.length()*diameter/250, (int) (curvePoint[1])+6);
							}
							else
							{
								final float dash1[] = {70.0f};
							    final BasicStroke dashed =
							        new BasicStroke(23.5f,
							                        BasicStroke.CAP_BUTT,
							                        BasicStroke.JOIN_MITER,
							                        50.0f, dash1, 30.0f);
								brush.setColor(Color.BLACK);
								brush.setStroke(dashed);//new BasicStroke(10)
								brush.draw(bezier);
							}
						}
					}
				}		
			}
		}
	}
	// --------------------------------------------------------------------------------------------
	public void paintRootUnitBody(Graphics2D brush, double panelScale, NAnimator Animator) 
	{
		if(Core.is(NVertex.MotherRoot)) 
		{
		   diameter = defaultDiameter/2;
		   if(position.length == 4) 
		   {   
			   brush.setColor(Color.cyan);
			   Ellipse2D.Double Ellipse = new Ellipse2D.Double(position[2] - (diameter*(subToSuperScale*0.9) + 40) / 2, position[3] - (diameter*(subToSuperScale*0.9) + 40) / 2, (diameter*(subToSuperScale*0.9)) + 40, (diameter*(subToSuperScale*0.9)) + 40);
			   Ellipse2D.Double Ellipse2 = new Ellipse2D.Double((position[2] - (diameter*subToSuperScale + 40) / 2), (position[3] - (diameter*subToSuperScale + 40) / 2), (diameter*subToSuperScale + 40), (diameter*subToSuperScale + 40));
			   Area clipArea = new Area(Ellipse2);
			   clipArea.subtract(new Area(Ellipse));
			   brush.setClip(clipArea);
			   brush.fill(Ellipse2);
			   brush.setClip(null);
			   
		       Paint old = brush.getPaint();
		       Point2D center = new Point2D.Float((float)position[2], (float)position[3]);
			   float gradientRadius = (float)((diameter*(subToSuperScale*0.9) + 40) / 2);   
			   float[] dist = { 0.3f, 0.9f, 1f };
			   Color[] colors = {new Color(0, 0, 0, 0.55f), new Color(0, 0, 0, 0.6f), Color.BLACK  };		
			   RadialGradientPaint rgp = new RadialGradientPaint(center, gradientRadius, dist, colors, CycleMethod.NO_CYCLE);
			   brush.setPaint(rgp);

			   brush.fill(Ellipse);
			   brush.setPaint(old);
			   //paintNodeDeck(brush, (int) position[2], (int) position[3], modifier, (int)(diameter*4.7 + 40) / 2, Animator);
		   }
		}
	}
	public void paintNeuron(Graphics2D brush, double panelScale, NAnimator Animator) 
	{
		int x = (int) position[0];
		int y = (int) position[1];
		brush.setColor(Color.cyan);
		double modifier = panelScaleModifier(panelScale);
		//COLORING FOR NEURON TYPES (FIRST, HIDDEN, LAST)
		//=================================
		if (Core.isHidden()) 
		{
			deckColor = new Color(0, 60, 120);
		}
		if (Core.isFirst()) 
		{
			deckColor = new Color(0, 140, 30);
		}
		if (Core.isLast()) 
		{
			deckColor = new Color(140, 0, 30);
		}
		if (Core.isLast() && Core.isFirst()) 
		{
			deckColor = new Color(140, 140, 0);
		}
		if (Core.isParalyzed()) 
		{
			deckColor = new Color(0, 0, 0);
		}
		//=================================

		if(Core.is(NVertex.MotherRoot)) 
		{
			diameter = defaultDiameter/2;
		}
		else if(Core.is(NVertex.BasicRoot))
		{
			Color color = NColor.SystemOcean;
			brush.setColor(color);
			diameter = defaultDiameter/4;
			deckColor = Color.BLACK;
			darkDeckColor = Color.BLACK;
			if(position.length == 4) 
			{
			    double[] old = position;
				position = new double[2];
				position[0] = old[0];
				position[1] = old[1];
			}
			else 
			{
				  
			}
		}
		else if(Core.is(NVertex.RootMemory))
		{
			Color color = NColor.SystemOcean;
			brush.setColor(color);
			diameter = defaultDiameter/3;
		}
		else
		{
			diameter = defaultDiameter;
		}
		//GHOST EFFECT:
		//--------------
		Point2D center = new Point2D.Float(x, y);
		float gradientRadius = (float) ((diameter / 2) + 80);
		float[] dist = { 0.0f, 0.5f, 1f };
		Color[] colors = { Color.cyan, Color.cyan, new Color(0, 0, 0, 0) };
		Paint oldPaint = brush.getPaint();
		RadialGradientPaint rgp = new RadialGradientPaint(center, gradientRadius, dist, colors, CycleMethod.NO_CYCLE);
		brush.setPaint(rgp);
		brush.fillOval(x - 100 - (int) ((diameter / 2)), y - 100 - (int) ((diameter / 2)), 
					(int) (diameter + 200),
					(int) (diameter + 200));
		brush.setPaint(oldPaint);
		//--------------
		
		//CYAN BASE:
		//--------------
		//brush.setColor(Color.cyan);
		brush.fillOval(x - (diameter + 40) / 2, y - (diameter + 40) / 2, diameter + 40, diameter + 40);


		//--------------
		for (int i = 0; i < InputNode.size(); i++) 
		{	
			//COLORING IF INPUT IS ROOT CONNECTION
			brush.setColor(Color.CYAN);
			if(getCore().asCore().getConnection()[i]!=null) 
			    {//System.out.println("node existing");
				 if(getCore().asCore().getConnection()[i][0].asCore().is(NVertex.Child)) 
				    {Color color = NColor.SystemOcean; brush.setColor(color);}
			    }
			else {if(getCore().asCore().is(NVertex.Child)) 
			         {Color color = NColor.SystemOcean; brush.setColor(color);}
			}
			InputNode.get(i).paint(brush, getX(), getY(), ((diameter / 2)));
		}
		//--------------
		
		//NEURON DECK:
		//--------------
		
		if (deckIsRemoved==true && deckAnimationRunning==false) {
			paintNodeInterior(brush, (int) position[0], (int) position[1], modifier, panelScale, Animator);
		}
		else if(deckIsRemoved==false && deckAnimationRunning==false) 
		{
			paintNodeDeck(brush, (int) position[0], (int) position[1], modifier, panelScale, Animator);
		}
		else if(deckAnimationRunning==true) 
		{
			paintNodeInterior(brush, (int) position[0], (int) position[1], modifier, (diameter/2), Animator);
			paintNodeDeck(brush, (int) position[0], (int) position[1], modifier, (diameter/2), Animator);
		}
		
		double mod = (panelScale/0.2);
		if(panelScale>0.1) 
		{
			DecimalFormat Formatter = new DecimalFormat("###.###E0");
			for (int i = 0; i < InputNode.size(); i++) 
			{
				InputNode.get(i)
				.paintTop(
					brush, 
					getX(), 
					getY(),
					diameter / 2, 
					Formatter.format(Core.getInput(0, 0)), 
					(Math.pow(mod, 6))
					);
			//InputField.get(i).
			}
		}
	
	}
	//NEURON DECK:
	// --------------------------------------------------------------------------------------------
	private void paintNodeInterior(Graphics2D brush, int centerX, int centerY, double modifier, double panelScale, NAnimator Animator) 
	{
		double radius = (double)diameter/2;
		int animationID   = 0;
		double buttonAnimationModifier = 0;
		int deckAnimationCounter = 0;

		animationID   = 0;
		if(Animator.hasCounter(this, animationID)) {buttonAnimationModifier = (double) Animator.getCounterOf(this, animationID) / buttonLimit;}
		//animationID   = 1;
		//if(Animator.hasCounter(this, animationID)) {}
		animationID   = 2;
		if(Animator.hasCounter(this, animationID)) 
		{
			deckAnimationCounter = Animator.getCounterOf(this, animationID);
		}
		
		Ellipse2D.Double ellipse = 
				new Ellipse2D.Double(
						centerX - (radius * modifier), centerY - (radius * modifier),
				        (2 * radius * modifier),       (2 * radius * modifier)
				);
		brush.setColor(Color.BLACK);
		brush.fill(ellipse);

		double decimal = 0;
		if(deckAnimationCounter!=0) 
		{
			decimal = ((double)(deckAnimationCounter/4)-deckAnimationLimit/8);
			//System.out.println(decimal);
			decimal /=(deckAnimationLimit/8);
			//System.out.println(decimal+"....gusto");
			if(decimal>1) {decimal=1;}else if(decimal<-1) {decimal=-1;}
			if(deckIsRemoved) {decimal*=-1;}
		}

		brush.setColor(deckButtonColor);
		Rectangle2D.Double rectangle = new Rectangle2D.Double(
				centerX - (buttonAnimationModifier * radius / 2 * Math.pow(modifier, 200 / modifier)) / 2,
				centerY - (buttonAnimationModifier * radius / 4 * Math.pow(modifier, 300 / modifier)) / 2,
				(buttonAnimationModifier * radius / 2 * Math.pow(modifier, 200 / modifier)),
				(buttonAnimationModifier * radius / 4 * Math.pow(modifier, 300 / modifier)));
		//brush.setClip(rectangle);
		brush.fillRoundRect(centerX - (int) (buttonAnimationModifier * radius/2 * Math.pow(modifier, 200 / modifier)) / 2,
				centerY - (int) (buttonAnimationModifier * radius / 4 * Math.pow(modifier, 300 / modifier)) / 2,
				(int) (buttonAnimationModifier * radius / 2 * Math.pow(modifier, 200 / modifier)),
				(int) (buttonAnimationModifier * radius / 4 * Math.pow(modifier, 300 / modifier)), (int)(radius/6), (int)(radius/6));
		//brush.setClip(null);
		Font NFont = new Font("Tahoma", Font.PLAIN, (int) (radius/9 * buttonAnimationModifier * Math.pow(modifier, 100 / modifier)));
		brush.setFont(NFont);
		double colorMod = buttonAnimationModifier * (Math.pow(modifier, 100 / modifier));
		if (colorMod > 1) {
			colorMod = 1;
		}
		if (colorMod < 0) {
			colorMod = 0;
		}
		brush.setColor(new Color((int) (deckColor.getRed() + (255 - deckColor.getRed()) * colorMod),
				(int) (deckColor.getGreen() + (255 - deckColor.getGreen()) * colorMod),
				(int) (deckColor.getBlue()  + (255 - deckColor.getBlue()) * colorMod)));
		brush.drawString("CLOSE", (int) (centerX - diameter*(0.09) * colorMod), (int) (centerY + diameter*(0.0154761) * colorMod));

		//III
		brush.setColor(Color.CYAN);
		brush.fillRoundRect((int)(centerX-radius*0.3), (int)(centerY-radius*0.1-radius*0.275*buttonAnimationModifier), (int)(radius*0.3*2), (int)(radius*2*0.1), (int)(radius*0.1), (int)(radius*0.5));
		Font ValueFont = new Font("Tahoma", Font.PLAIN, (int)(diameter*(0.063333)));
		brush.setFont(ValueFont);
		DecimalFormat Formatter = new DecimalFormat("###.###E0");
		
		brush.drawString("Activation", (int) (centerX - radius*0.27), (int) (centerY - radius*0.1 - radius*0.27500*buttonAnimationModifier));
		brush.setColor(Color.BLACK);
		brush.drawString(Formatter.format(Core.getActivation(0)), (int) (centerX - radius*0.108), (int) (centerY +radius*0.0475- radius*0.260*buttonAnimationModifier));
		
		ValueFont = new Font("Tahoma", Font.PLAIN, (int)(diameter*(0.033333)));
		brush.setFont(ValueFont);
		
		brush.setColor(Color.CYAN);
		brush.drawString("Paralyzed: "+Core.isParalyzed()+";", 
				       (int) (centerX - radius*0.2), (int) (centerY -radius*0.775));
		brush.setColor(Color.DARK_GRAY);
		brush.fillRect((int) (centerX - radius*0.4), (int) (centerY -radius*0.77), (int)(2*radius*0.4), (int)(radius*0.01));
		
		brush.setColor(Color.CYAN);
		brush.drawString("First: "+Core.isFirst()+"; Hidden: "+Core.isHidden()+"; Last: "+Core.isLast()+";", 
					   (int) (centerX - radius*0.5),  (int) (centerY -radius*0.65));
		brush.setColor(Color.DARK_GRAY);
		brush.fillRect((int) (centerX - radius*0.625), (int) (centerY -radius*0.645), (int)(2*radius*0.625), (int)(radius*0.01));
		
		String rootType = "";
		if(Core.is(NVertex.Root)) 
		{
			if(Core.is(NVertex.TrainableBasicRoot)) {rootType += "trainable ";}
			if(Core.is(NVertex.BasicRoot))  	    {rootType += "basic ";}
			if(Core.is(NVertex.RootMemory)) 		{rootType += "memory ";}
			if(Core.is(NVertex.RootInput))  		{rootType += "input ";}
			if(Core.is(NVertex.MotherRoot))  		{rootType += "super ";}
			
			if(rootType.isEmpty()==false) {rootType = " ( "+rootType+")";}
		}
		brush.setColor(Color.CYAN);
		brush.drawString("Is root component: "+Core.is(NVertex.Root)+rootType+";", 
				       (int) (centerX - radius*0.65), (int) (centerY -radius*0.520));
		brush.setColor(Color.DARK_GRAY);
		brush.fillRect((int) (centerX - radius*0.675), (int) (centerY -radius*0.515), (int)(2*radius*0.675), (int)(radius*0.01));
		
		brush.setColor(Color.CYAN);
		brush.drawString("Gradient behavior",                            (int) (centerX - radius*0.85), (int) (centerY -radius*0.1));
		brush.drawString("AutoApply: "+Core.isInstantGradientApply()+";", (int) (centerX - radius*0.825), (int) (centerY -radius*0.0));
		brush.drawString("Summing: "+Core.isGradientSumming()+";",        (int) (centerX - radius*0.825), (int) (centerY +radius*0.1));
		brush.setColor(Color.DARK_GRAY);
		brush.fillRect((int) (centerX - radius*0.9), (int) (centerY +radius*0.15), (int)(2*radius*0.9), (int)(radius*0.01));
		
		brush.setColor(Color.CYAN);
		brush.drawString("Trainable",								 (int) (centerX + radius*0.425), (int) (centerY -radius*0.1));
		brush.drawString("Bias: "+Core.biasIsTrainable()+"; ",    (int) (centerX + radius*0.425), (int) (centerY -radius*0.0));
		brush.drawString("Weight: "+Core.asCore().weightIsTrainable()+";",  (int) (centerX + radius*0.4), (int) (centerY +radius*0.1));
		
		brush.setColor(Color.CYAN);
		brush.drawString("Neuron ID",	(int) (centerX - radius*0.14), (int) (centerY +radius*0.225));
		ValueFont = new Font("Tahoma", Font.PLAIN, (int)(diameter*(0.065)));
		brush.setFont(ValueFont);
		brush.setColor(Color.DARK_GRAY);
		brush.fillRect((int) (centerX - radius*0.1*Long.toString(Core.getID()).length()), (int) (centerY +radius*0.385), (int)(2*radius*0.1*Long.toString(Core.getID()).length()), (int)(radius*0.02));

		brush.setColor(Color.CYAN);
		brush.drawString(Long.toString(Core.getID()),(int) (centerX - radius*0.0297*Long.toString(Core.getID()).length()), (int) (centerY +radius*0.385));
		//brush.drawString(Long.toString(Core.getID()),(int) (centerX - radius*0.02927*Long.toString(Core.getID()).length()), (int) (centerY +radius*0.385));
		
		ValueFont = new Font("Tahoma", Font.PLAIN, (int)(diameter*(0.033333)));
		brush.setFont(ValueFont);
		
		brush.setColor(Color.CYAN);
		brush.drawString("Property Compartments",	(int) (centerX - radius*0.325), (int) (centerY +radius*0.475));
	
		brush.setColor(Color.DARK_GRAY);
		brush.fillRect((int) (centerX - radius*0.5), (int) (centerY +radius*0.49), (int)(2*radius*0.5), (int)(radius*0.01));
		ValueFont = new Font("Tahoma", Font.PLAIN, (int)(diameter*(0.025)));
		
		brush.setFont(ValueFont);
		brush.setColor(Color.CYAN);
		String listedProperties = "";
		boolean somethingToList = true;
		double YMod = - radius*0.05;
		
		NVCloak Public = (NVCloak) Core.findModule(NVCloak.class);
		if(Core.getFunction()!=null) 
		{
			listedProperties += Core.getFunction().expression()+"";
		}
		NVMemory Memory = (NVMemory) Core.findModule(NVMemory.class);
		if(Memory!=null) 
		{
			listedProperties += "Memory";
			if(Memory.hasActivationMemory()) {listedProperties += " (Activation";}
			if(Memory.hasActivityMemory()) {listedProperties += ", Activity";}
			if(Memory.hasDeriviationMemory()) {listedProperties += ", Derivatives)";}else {listedProperties += ");";}
		}
		brush.drawString(listedProperties, (int)(centerX - radius*0.013*listedProperties.length()), (int)(centerY +radius*0.6+YMod));
		YMod+=radius*0.075;listedProperties="";
			
		double[][] nodeResult = null;
		if(Public!=null) 
		{
			nodeResult = Public.getPublicData();
		}
		listedProperties += "Public Values ";
		if(nodeResult==null) 
		{
			listedProperties+=" (null)";
		}
		else 
		{
			if(nodeResult.length==1) {listedProperties+="(Activation:"+nodeResult[0]+")";}
			if(nodeResult.length==2) {listedProperties+="(Activation:"+nodeResult[0]+", Error:"+nodeResult[1]+")";}
			if(nodeResult.length==3) {listedProperties+="(Activation:"+nodeResult[0]+", Error:"+nodeResult[1]+", Optimum:"+nodeResult[2]+")";}
		}
		brush.drawString(listedProperties,(int) (centerX - radius*0.013*listedProperties.length()), (int) (centerY +radius*0.6+YMod));
		YMod+=radius*0.075;listedProperties="";
		//III
		brush.setColor(Color.CYAN);
		
		if (Core.isLast()) 
		{
			brush.drawString("Optimum:", (int) (centerX - 50), (int) (centerY - 240));
			brush.drawString(""+Formatter.format(Core.getOptimum(0)), (int) (centerX - 50), (int) (centerY - 210));
		}

		ArrayList<Painter> PainterArray = new ArrayList<Painter>();
		if(Core.asCore().hasModule(NVControlFrame.class))
		{
			PainterArray.add
			((Graphics2D painterBrush, double x, double y)->
			{
			   painterBrush.setColor(Color.DARK_GRAY);
			   painterBrush.fillOval((int)(x-radius*0.1), (int)(y-radius*0.1), (int)((radius*0.1)*2), (int)((radius*0.1)*2));
			});
		}
			
	}
	
	private void paintNodeDeck(Graphics2D brush, int centerX, int centerY, double modifier, double panelScale, NAnimator Animator) 
	{
		double radius = (double)diameter/2;
		//int centerX = (int) position[0];
		//int centerY = (int) position[1];
		int animationID   = 0;
		double buttonAnimationModifier = 0;
		int deckAnimationCounter = 0;
		
		animationID   = 0;
		if(Animator.hasCounter(this, animationID)) 
		{
			buttonAnimationModifier = (double) Animator.getCounterOf(this, animationID) / buttonLimit;
		}
		//animationID   = 1;
		//if(Animator.hasCounter(this, animationID)) {}
		animationID   = 2;
		if(Animator.hasCounter(this, animationID)) 
		{
			deckAnimationCounter = Animator.getCounterOf(this, animationID);
		}
		
		//System.out.println(deckAnimationCounter+"....counter");
		double decimal = 0;
		if(deckAnimationCounter!=0) {
			decimal = ((double)(deckAnimationCounter/4)-deckAnimationLimit/8);
			//System.out.println(decimal);
			
			decimal /=(deckAnimationLimit/8);
			
			//System.out.println(decimal+"....gusto");
			if(decimal>1) 
			{
				decimal=1;
			}
			else if(decimal<-1) 
			{
				decimal=-1;
			}
			if(deckIsRemoved) 
			{
				decimal*=-1;
			}
			Area clipArea = new Area(new Rectangle.Double(centerX-radius, centerY-radius, radius*2, radius*2));
			clipArea.subtract(new Area(new Ellipse2D.Double(centerX-radius*decimal, centerY-radius*decimal, radius*2*decimal, radius*2*decimal)));
			brush.setClip(clipArea);
		}
		
		//System.out.println("ButtonAnimationModifier: "+buttonAnimationModifier);
		//GRADIENT EFFECT:
		Point2D center = new Point2D.Float(centerX, centerY);
		float gradientRadius = (float) (radius + 50);
		float[] dist = { 0.0f, 0.6f, 1f }; 
		if(decimal>0) 
		{
			dist[1]+=(float)(0.399f*decimal);
		}
		Color[] colors = { deckColor, darkDeckColor, Color.black };
		Paint oldPaint = brush.getPaint();
		RadialGradientPaint rgp = new RadialGradientPaint(center, gradientRadius, dist, colors, CycleMethod.NO_CYCLE);
		brush.setPaint(rgp);

		Ellipse2D.Double ellipse = new Ellipse2D.Double(centerX - (radius * modifier), centerY - (radius * modifier),
														(2*radius * modifier), (2*radius * modifier));

		brush.fill(ellipse);
		brush.setPaint(oldPaint);
		//brush.setClip(ellipse);
		//brush.setClip(null);

		if(Core.is(NVertex.BasicRoot)) 
		{
		    brush.setColor(Color.CYAN);
			brush.fillRoundRect( centerX - (int)(radius * 0.75), centerY - (int) (radius / 14), (int) (radius*1.5), (int) (radius/7), (int) (radius*0.4), (int) (radius*0.4));
			brush.rotate(1*Math.PI*0.25, centerX, centerY);
			brush.fillRoundRect( centerX - (int)(radius * 0.75), centerY - (int) (radius / 14), (int) (radius*1.5), (int) (radius/7), (int) (radius*0.4), (int) (radius*0.4));
			brush.rotate(1*Math.PI*0.25, centerX, centerY);
			brush.fillRoundRect( centerX - (int)(radius * 0.75), centerY - (int) (radius / 14), (int) (radius*1.5), (int) (radius/7), (int) (radius*0.4), (int) (radius*0.4));
			brush.rotate(1*Math.PI*0.25, centerX, centerY);
			brush.fillRoundRect( centerX - (int)(radius * 0.75), centerY - (int) (radius / 14), (int) (radius*1.5), (int) (radius/7), (int) (radius*0.4), (int) (radius*0.4));
			brush.rotate(-3*Math.PI*0.25, centerX, centerY);
		}
		Font NFont;
		NFont = new Font("Tahoma", Font.BOLD, (int)(diameter*(0.05)));
		brush.setFont(NFont);
		double transparencyMod = 1;//System.out.println(panelScale);
		if(panelScale<0.2) 
		{
			transparencyMod = panelScale/0.2;
		}
		//System.out.println(transparencyMod);
		int transparency = (int)(transparencyMod*transparencyMod*transparencyMod*255);
		//System.out.println(transparency);
		
		brush.setColor(darkDeckColor);
		ellipse = new Ellipse2D.Double(
				centerX - (buttonAnimationModifier * radius * Math.pow(modifier, 200 / modifier)) / 2,
				centerY - (buttonAnimationModifier * radius / 1.9 * Math.pow(modifier, 300 / modifier)) / 2,
				          (buttonAnimationModifier * radius * Math.pow(modifier, 200 / modifier)),
				          (buttonAnimationModifier * radius / 1.9 * Math.pow(modifier, 300 / modifier)));
		//brush.setClip(ellipse);
		brush.fill(ellipse);
		//brush.setClip(null);

		brush.setColor(deckButtonColor);
		Rectangle2D.Double rectangle = new Rectangle2D.Double(
				centerX - (buttonAnimationModifier * radius / 2 * Math.pow(modifier, 200 / modifier)) / 2,
				centerY - (buttonAnimationModifier * radius / 4 * Math.pow(modifier, 300 / modifier)) / 2,
				(buttonAnimationModifier * radius / 2 * Math.pow(modifier, 200 / modifier)),
				(buttonAnimationModifier * radius / 4 * Math.pow(modifier, 300 / modifier)));
		
		if(Core.is(NVertex.BasicRoot)) 
		{
			brush.setColor(Color.CYAN);
			brush.fillRoundRect((int)(centerX-radius*0.3), 
					            (int)(centerY-radius*0.22-radius*0.275*buttonAnimationModifier), 
					            (int)(radius*0.3*2), (int)(radius*2*0.1), (int)(radius*0.1), (int)(radius*0.5));
			
			brush.setColor(deckButtonColor);
		}
		
		//brush.setClip(rectangle);
		brush.fillRoundRect(centerX - (int) (buttonAnimationModifier * radius/2 * Math.pow(modifier, 200 / modifier)) / 2,
				centerY - (int) (buttonAnimationModifier * radius / 4 * Math.pow(modifier, 300 / modifier)) / 2,
				(int) (buttonAnimationModifier * radius / 2 * Math.pow(modifier, 200 / modifier)),
				(int) (buttonAnimationModifier * radius / 4 * Math.pow(modifier, 300 / modifier)), (int)(radius/6), (int)(radius/6));
		//brush.setClip(null);
		
		DecimalFormat Formatter = new DecimalFormat("###.###E0");
		Font ValueFont = new Font("Tahoma", Font.BOLD, (int)(diameter*(0.065)));
		brush.setFont(ValueFont);
        //System.out.println(transparency);
		if(transparency>35) {
		brush.setColor(new Color(0,0,0,transparency));
		brush.fillRect((int) (centerX - radius*0.1*Long.toString(Core.getID()).length()), (int) (centerY -radius*0.25), (int)(2*radius*0.1*Long.toString(Core.getID()).length()), (int)(radius*0.02));
		//brush.setColor(Color.BLACK);
		brush.drawString(Long.toString(Core.getID()),(int) (centerX - radius*0.038*Long.toString(Core.getID()).length()), (int) (centerY -radius*0.25));//+radius*0.385
		//brush.drawString(Long.toString(Core.getID()),(int) (centerX - radius*0.0297*Long.toString(Core.getID()).length()), (int) (centerY +radius*0.385));
		ValueFont = new Font("Tahoma", Font.PLAIN, (int)(diameter*(0.063333)));
		brush.setFont(ValueFont);

		brush.drawString("Activation", (int) (centerX - radius*0.27), (int) (centerY - radius*0.1 - radius*0.27500*buttonAnimationModifier));
		
		}
		
		double expMod = 1;
		expMod = ((double)diameter/defaultDiameter)*panelScale/0.15;
			//System.out.println( (diameter/defaultDiameter));
		if(expMod>0.7 && Core.getFunction()!=null) 
		{
			if(expMod>1) {expMod=1;}
			//System.out.println(panelScale);
			//System.out.println(transparencyMod);
			int expTrans = (int)(Math.pow(expMod, 10)*255);
			//System.out.println(transparency);
			String exp = Core.getFunction().expression();
			
			RoundRectangle2D.Double rec = 
			new RoundRectangle2D.Double();
			
			rec.setRoundRect(
					(centerX - radius*0.033*exp.length()), 
					(centerY +radius*0.08), 
					(2*radius*0.033*exp.length()), 
					(radius*0.19), 
					 radius*0.13, 
					 radius*0.13);
			
			brush.setFont(new Font("Tahoma", Font.BOLD, (int)(diameter*(0.04))));
			brush.setColor(new Color(0,0,0,expTrans));
			brush.fill(rec);
			rec.setRoundRect(
					(centerX - radius*0.031*exp.length()), 
					(centerY +radius*0.1), 
					(2*radius*0.031*exp.length()), 
					(radius*0.15), 
					 radius*0.1, 
					 radius*0.1);
			brush.setColor(new Color(NColor.SystemOcean.getRed(),NColor.SystemOcean.getGreen(),NColor.SystemOcean.getBlue(),expTrans));
			
			brush.fill(rec);
			
			//brush.setColor(Color.BLACK);
			brush.setColor(new Color(0,0,0,expTrans));
			brush.drawString((exp),
					         (int) (centerX - radius*0.02*exp.length()), 
					         (int) (centerY +radius*0.2));
		}
		NFont = new Font("Tahoma", Font.PLAIN, (int) (radius/9 * buttonAnimationModifier * Math.pow(modifier, 100 / modifier)));
		brush.setFont(NFont);
		double colorMod = buttonAnimationModifier * (Math.pow(modifier, 100 / modifier));
		if (colorMod > 1) 
		{
			colorMod = 1;
		}
		if (colorMod < 0) 
		{
			colorMod = 0;
		}
		brush.setColor(new Color((int) (deckColor.getRed() + (255 - deckColor.getRed()) * colorMod),
				(int) (deckColor.getGreen() + (255 - deckColor.getGreen()) * colorMod),
				(int) (deckColor.getBlue()  + (255 - deckColor.getBlue()) * colorMod)));
		brush.drawString("OPEN", (int) (centerX - diameter*(0.08) * colorMod), (int) (centerY + diameter*(0.0154761) * colorMod));

		double Amod = 1;
		if(panelScale<0.3) {Amod=1;}
		//ACTIVATION BOX:
		brush.setColor(Color.BLACK);
		brush.fillRoundRect(
				(int)(centerX-radius*0.4*Amod), 
				(int)(centerY-radius*0.1*Amod-radius*0.275*buttonAnimationModifier), 
				(int)(radius*0.4*2*Amod), 
				(int)(radius*2*0.1*Amod), 
				(int)(radius*0.1*Amod), 
				(int)(radius*0.5*Amod));
		
		brush.setColor(Color.cyan);
		ValueFont = new Font("Tahoma", Font.PLAIN, (int)(diameter*(0.08)));
		brush.setFont(ValueFont);
		String acti = Formatter.format(Core.getActivation(0));
		brush.drawString(acti, 
				(int) (centerX - radius*0.043*acti.length()*Amod), 
				(int) (centerY + radius*0.055- radius*0.275*buttonAnimationModifier*Amod));
		
		ValueFont = new Font("Tahoma", Font.PLAIN, (int)(diameter*(0.033333)));
		brush.setFont(ValueFont);
		
		if (Core.isLast()) 
		{
			brush.drawString("Optimum:", (int) (centerX - 50), (int) (centerY - 240));
			brush.drawString(""+Formatter.format(Core.getOptimum(0)), (int) (centerX - 25), (int) (centerY - 210));
		}
		brush.setClip(null);
	}

	// --------------------------------------------------------------------------------------------
	private double panelScaleModifier(double currentPanelScale) 
	{
		if (5 * currentPanelScale < 1) 
		{
			return Math.pow((1 - (1 - currentPanelScale * 5) / 35), 7);
		} 
		else 
		{
			return 1;
		}
	}
	// --------------------------------------------------------------------------------------------
	public boolean needsPanelUpdate() 
	{
		return false;
	}
	// --------------------------------------------------------------------------------------------
	public double[] getNeuronSpace() //IS THIS IN USE? / NECESSARRY?
	{
		double[] rect = new double[4];
		rect[0] = position[0] - diameter / 2;
		rect[1] = position[1] - diameter / 2;
		rect[2] = position[0] + diameter / 2;
		rect[3] = position[1] + diameter / 2;
		return rect;
	}
	// --------------------------------------------------------------------------------------------
	public double getUnitRadius() 
	{
		return (double) (diameter) / 2;
	}
	// --------------------------------------------------------------------------------------------
	public double getX() 
	{
		return position[0];
	}
	// --------------------------------------------------------------------------------------------
	public double getY() 
	{
		return position[1];
	}
	// --------------------------------------------------------------------------------------------
	public void addToX(double value) 
	{
		position[0] += value;
	}
	// --------------------------------------------------------------------------------------------
	public void addToY(double value) 
	{
		position[1] += value;
	}
	// --------------------------------------------------------------------------------------------
	//NPanelObject implementation:
	@Override
	public double getLeftPeripheral() 
	{
		if(position.length==4) {
			double superPeripheral = position[2]-diameter/2*subToSuperScale;
			double subPeripheral   = position[0]-diameter/2; 
			return Math.min(superPeripheral, subPeripheral);
		}
		else {
			return position[0]-diameter/2; 
		}
	}
	@Override
	public double getTopPeripheral() {
		if(position.length==4) {
			double superPeripheral = position[3]-diameter/2*subToSuperScale;
			double subPeripheral   = position[1]-diameter/2; 
			return Math.min(superPeripheral, subPeripheral);
		}
		else {
			return position[1]-diameter/2; 
		}
	}
	@Override
	public double getRightPeripheral() 
	{
		if(position.length==4) 
		{
			double superPeripheral = position[2]+diameter/2*subToSuperScale;
			double subPeripheral   = position[0]+diameter/2; 
			return Math.max(superPeripheral, subPeripheral);
		}
		else 
		{
			return position[0]+diameter/2; 
		}
	}
	@Override
	public double getBottomPeripheral() 
	{
		if(position.length==4) 
		{
			double superPeripheral = position[3]+diameter/2*subToSuperScale;
			double subPeripheral   = position[1]+diameter/2; 
			return Math.max(superPeripheral, subPeripheral);
		}
		else 
		{
			return position[1]+diameter/2; 
		}
	}
	@Override
	public int getLayerID() 
	{
		int layerID = 3;
		if(Core.is(NVertex.MotherRoot)) 
		{
			return layerID;
		}
		layerID++;
		if(Core.is(NVertex.BasicRoot)||Core.is(NVertex.RootInput)) 
		{
			return layerID;
		}
		layerID++;
		if(Core.is(NVertex.RootMemory))
		{
			return layerID;
		}
		return layerID;
	}
	@Override
	public boolean needsRepaintOnLayer(int layerID) 
	{
		if(layerID==0) 
		{
			if(Core.is(NVertex.Root)==false) 
			{
				return true;
			}
			return false;
		}
		if(layerID==1) 
		{
			if(Core.is(NVertex.MotherRoot)) 
			{
				return true;
			}
			return false;
		}
		if(layerID==2) 
		{
			if(Core.is(NVertex.Root)==true) 
			{
				return true;
			}
			return false;
		}
		int mainLayerID = getLayerID();
		if(mainLayerID==layerID)
		{
			return true;
		}
		return false;
	}
	@Override
	public void repaintLayer(int layerID, Graphics2D brush, NPanelAPI HostSurface) 
	{
		switch(layerID) 
		{
			case 0://Paint Connections
				paintNodeConnections(brush, HostSurface.getAnimator());
				break;
			case 1://Paint root body
				paintRootUnitBody(brush, HostSurface.getScale(), HostSurface.getAnimator());
				break;
			case 2://Paint timeless root connections
				paintNodeConnections(brush, HostSurface.getAnimator());
				break;
			case 3://Paint basic root nodes
				paintNeuron(brush, HostSurface.getScale(), HostSurface.getAnimator());
				break;
			case 4://Paint memory root nodes
				paintNeuron(brush, HostSurface.getScale(), HostSurface.getAnimator());
				break;
			case 5://Paint super root nodes
				paintNeuron(brush, HostSurface.getScale(), HostSurface.getAnimator());
				break;
			case 6://Paint default nodes
				paintNeuron(brush, HostSurface.getScale(), HostSurface.getAnimator());
				break;
		}
	}
	@Override
	public double getRadius() 
	{
		return diameter/2;
	}
	@Override
	public boolean hasGripAt(double x, double y, NPanelAPI HostPanel) 
	{
		if(this.testForBody(x, y)) 
		{
			return true;
		}
		if(this.testForBody_Root(x, y)) 
		{
			return true;
		}
		return false;
	}
	@Override
	public boolean killable() 
	{
		return false;
	}


}
