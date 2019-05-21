package neureka.ngui;

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
import java.awt.geom.Point2D;
import java.util.ArrayList;

import neureka.main.core.NVertex;

public class NCircleMenu implements NPanelObject{

	private final static int ReferenceRadius      =300;
	private final static int InnerReferenceRadius =100;
	private final static long CounterLimit        =450000000L;

	private int layerID = 6;
	
	private double X;
	private double Y;
	private int highlightID = 2;
	private boolean needsRepaint;
	private int baseMenuChoice = 0;
	
	private boolean isDying = false;
	private float rotation=((float)(Math.PI));
	
	private double animationCounter=-CounterLimit*4;
	private byte[] mapping;
	private boolean[] checked;
	private double birthScale;
	
	private Object PropertyContainer;
	//private NPanel HostPanel;
	
	NGraphBuilder Builder;
	
	private MenuActor Actor = 
	(int id, Object thing)->
	{
		System.out.println("Actor has not been e_set to do anything :/ ");
	};
	
	Color SystemOcean  = new Color(0,160,210);
	Color checkedGreen = new Color(10,255,100);
	Color neuronBlue = new Color(40, 100, 255);
	//-----------------------------------------------------------------------------------
	NCircleMenu(double centerX, double centerY, double scale, NGraphBuilder builder)
	{
		X=centerX;
		Y=centerY;
		birthScale=scale*2;
		Builder = builder;
		PropertyContainer = Builder.getSurface().findAnything(centerX, centerX, true, this);
	}
	//----------------------------------------------------------------------------------------------
	public void setLayerID(int id) 
	{
		System.out.println("e_set layerID...");
		layerID=id;
	}
	public void drawMenu(Graphics2D brush) 
	{
		double radius=ReferenceRadius/birthScale;
		double innerRadius=InnerReferenceRadius/birthScale;
		double primMod= ((double)animationCounter/CounterLimit);
		if(primMod<=0) {primMod=0.0;}
		double secMod = ((double)(animationCounter+CounterLimit*4)/(CounterLimit*4));
		if(secMod<=0) {secMod=0.0;}
		if(secMod>1) {secMod=1;}
		
		Area clipArea = new Area();
		clipArea = new Area(new Rectangle.Double((X-radius), (Y-radius), (2*radius), (2*radius)));
		clipArea.subtract(new Area(new Ellipse2D.Double((X-innerRadius*0.7), (Y-innerRadius*0.7),(2*innerRadius*0.7), (2*innerRadius*0.7))));
		brush.setClip(clipArea);
		int transparency = (int)(255*(secMod));
		//Math.abs(transparency);
		if(transparency<0) {transparency=0;}
		if(transparency>255) {transparency=255;}
		brush.setColor(new Color(0, 200,255, (int)(transparency)));
		brush.fillOval((int)(X-innerRadius), (int)(Y-innerRadius),(int)(2*innerRadius), (int)(2*innerRadius));
		//System.out.println("ho transparency: "+transparency);
		brush.setClip(null);
		
		if(mapping==null) {return;}
		
		if(primMod>0) {
			clipArea.add(new Area(new Ellipse2D.Double(X-radius, Y-radius, radius*2, radius*2)));
			//clipArea.subtract(new Area(new Ellipse2D.Double(X-innerRadius, Y-innerRadius, innerRadius*2, innerRadius*2)));
			brush.setClip(clipArea);
			Point2D center = new Point2D.Float((float)X, (float)Y);
				float gradientRadius = (float) (radius*primMod);
				float[] dist = { (float)0.3f, 0.35f, 1f };//(innerRadius)/radius)
				Color[] colors = { new Color(0, 0, 0, 0), Color.CYAN, new Color(0, 0, 0, 0) };
				Paint oldPaint = brush.getPaint();
				RadialGradientPaint rgp = new RadialGradientPaint(center, gradientRadius, dist, colors, CycleMethod.NO_CYCLE);
			brush.setPaint(rgp);
			brush.fillOval((int)(X-radius*primMod), (int)(Y-radius*primMod), (int)(radius*2*primMod), (int)(radius*2*primMod));
		 
			brush.setClip(null);
			double alpha = 2*Math.PI/mapping.length;
			double alphaMod = primMod*2*Math.PI;
			double rotationX=Math.cos(rotation)*0-Math.sin(rotation)*-1;
			double rotationY=Math.sin(rotation)*0+Math.cos(rotation)*-1;
			double newRotationX = Math.cos(alphaMod-alpha)*rotationX-Math.sin(alphaMod-alpha)*rotationY;
			double newRotationY = Math.sin(alphaMod-alpha)*rotationX+Math.cos(alphaMod-alpha)*rotationY;
			rotationX = newRotationX;
			rotationY = newRotationY; 
			brush.setColor(Color.magenta);
			for(int i=0; i<mapping.length; i++) {
				newRotationX = Math.cos(alpha)*rotationX-Math.sin(alpha)*rotationY;
				newRotationY = Math.sin(alpha)*rotationX+Math.cos(alpha)*rotationY;
				rotationX = newRotationX;
				rotationY = newRotationY; 
				if((Math.abs(rotationX) + Math.abs(rotationY)) != 1) 
			    	{
						double correction = 2-Math.pow(Math.pow(rotationX, 2) + Math.pow(rotationY, 2), 0.5);
						rotationX *= correction;
						rotationY *= correction;
			    	}
				if(i==mapping.length-1) {
					newRotationX = Math.cos(alpha)*rotationX-Math.sin(alpha)*rotationY;
					newRotationY = Math.sin(alpha)*rotationX+Math.cos(alpha)*rotationY;
					double clipRadius = primMod*(radius-innerRadius)/2;
					double clipX =(X)+newRotationX*(clipRadius+innerRadius)*(primMod);
					double clipY =(Y)+newRotationY*(clipRadius+innerRadius)*(primMod);
					clipArea.add(new Area(new Rectangle((int)(X-radius), (int)(Y-radius), (int)(radius*2), (int)(radius*2))));
					clipArea.subtract(new Area(new Ellipse2D.Double(clipX-clipRadius*0.99, clipY-clipRadius*0.99, 2*clipRadius*0.98, 2*clipRadius*0.98)));
					brush.setClip(clipArea);
				   }
				double ovalRadius = primMod*(radius-innerRadius)/2;
				double ovalX =(X)+rotationX*(ovalRadius+innerRadius)*(primMod);
				double ovalY =(Y)+rotationY*(ovalRadius+innerRadius)*(primMod);
				brush.fillRect((int)ovalX-10, (int)ovalY-10, 20, 20);
				center = new Point2D.Float((float)ovalX, (float)ovalY);
				gradientRadius = (float) ((radius*primMod-innerRadius*primMod)/2);
				dist[0]=0.8f;
				dist[1]=0.9f;
				dist[2]=1f;
			    colors[0]= new Color(10, 130, 255);
			    if(checked!=null) {if(checked.length>i) {if(checked[i]) {colors[1]=checkedGreen;}else {}
			    }else {colors[1]= new Color(5, 70, 255);}}else {System.out.println("Checked is null!!!!! = Bad");}
			    
			    colors[2]=new Color(0, 0, 0, 0);
				oldPaint = brush.getPaint();
				 rgp = new RadialGradientPaint(center, gradientRadius, dist, colors, CycleMethod.NO_CYCLE);
				 brush.setColor(Color.black);
				 brush.fillOval((int)(ovalX-ovalRadius), (int)(ovalY-ovalRadius), (int)(2*ovalRadius), (int)(2*ovalRadius));
				 brush.setPaint(rgp);
				 brush.fillOval((int)(ovalX-ovalRadius*0.99), (int)(ovalY-ovalRadius*0.99), (int)(2*ovalRadius*0.98), (int)(2*ovalRadius*0.98));
		 
				 drawMenuIcon(mapping[i],ovalX, ovalY, brush);
			}
			brush.setClip(null);
			brush.setColor(Color.CYAN);
			brush.fillRect((int)(X-10), (int)(Y-10), 20, 20);
			//paint chosen:
			if(highlightID!=0) {
				 alpha *=(highlightID);
				 newRotationX = Math.cos(alpha)*rotationX-Math.sin(alpha)*rotationY;
			     newRotationY = Math.sin(alpha)*rotationX+Math.cos(alpha)*rotationY;
			     rotationX = newRotationX;
				 rotationY = newRotationY; 
				 if((Math.abs(rotationX) + Math.abs(rotationY)) != 1) 
				    {
					 	double correction = 2-Math.pow(Math.pow(rotationX, 2) + Math.pow(rotationY, 2), 0.5);
					 	rotationX *= correction;
					 	rotationY *= correction;
				    }
				 double ovalRadius = primMod*(radius-innerRadius)/2;
				 double ovalX =(X)+rotationX*(ovalRadius+innerRadius)*(primMod);
				 double ovalY =(Y)+rotationY*(ovalRadius+innerRadius)*(primMod);
				 center = new Point2D.Float((float)ovalX, (float)ovalY);
					gradientRadius = (float) ((radius-innerRadius)/2);
					dist[0]=0.8f;
					dist[1]=0.9f;
					dist[2]=1f;
					colors[0]= Color.cyan;
				    colors[1]= Color.blue;
				    colors[2]=new Color(0, 0, 0, 0);
				    oldPaint = brush.getPaint();
					rgp = new RadialGradientPaint(center, gradientRadius, dist, colors, CycleMethod.NO_CYCLE);
				 brush.setColor(Color.black);
				 brush.fillOval((int)(ovalX-ovalRadius), (int)(ovalY-ovalRadius), (int)(ovalRadius*2), (int)(ovalRadius*2));
				 brush.setPaint(rgp);
				 brush.fillOval((int)(ovalX-ovalRadius*0.99), (int)(ovalY-ovalRadius*0.99), (int)(2*ovalRadius*0.98), (int)(2*ovalRadius*0.98));
				 drawMenuIcon(mapping[highlightID-1],ovalX, ovalY, brush);
			}
		 }		
	}
	public double[] getButtonCenterOfChoice(int choice) 
	{
		if(mapping==null || choice<=0) {return null;}
		double radius=ReferenceRadius/birthScale;
		double innerRadius=InnerReferenceRadius/birthScale;
		double primMod= ((double)animationCounter/CounterLimit);
		if(primMod<=0) {primMod=0.0;}
		double alpha = 2*Math.PI/mapping.length;
		double alphaMod = primMod*2*Math.PI;
		double rotationX=Math.cos(rotation)*0-Math.sin(rotation)*-1;
		double rotationY=Math.sin(rotation)*0+Math.cos(rotation)*-1;
		double newRotationX = Math.cos(alphaMod-alpha)*rotationX-Math.sin(alphaMod-alpha)*rotationY;
		double newRotationY = Math.sin(alphaMod-alpha)*rotationX+Math.cos(alphaMod-alpha)*rotationY;
		rotationX = newRotationX;
		rotationY = newRotationY; 
		
		newRotationX = Math.cos(alpha*choice)*rotationX-Math.sin(alpha*choice)*rotationY;
		newRotationY = Math.sin(alpha*choice)*rotationX+Math.cos(alpha*choice)*rotationY;
		rotationX = newRotationX;
		rotationY = newRotationY; 
		if((Math.abs(rotationX) + Math.abs(rotationY)) != 1) 
		{
			double correction = 2-Math.pow(Math.pow(rotationX, 2) + Math.pow(rotationY, 2), 0.5);
			rotationX *= correction;
			rotationY *= correction;
		}
		double ovalRadius = primMod*(radius-innerRadius)/2;
		double ovalX =(X)+rotationX*(ovalRadius+innerRadius)*(primMod);
		double ovalY =(Y)+rotationY*(ovalRadius+innerRadius)*(primMod);
			
		double[] choiceCenter = {ovalX, ovalY};
		return choiceCenter;
	}
	public double animationState() {return animationCounter;}
	public void modifyAnimationState(int value) {animationCounter+=value; }//if(animationCounter<=(-4*CounterLimit)) {animationCounter = -4*CounterLimit;}
	public boolean killable() {if(isDying) {if(animationCounter<=(-4*CounterLimit)) {return true;}}return false;}
	public double intrinsicScale() {return birthScale;}
	public double getX() {return X;}
	public double getY() {return Y;}
	//------------------------------------------------------------------------------------
	@Override
	public ArrayList<NPanelRepaintSpace> updateOn(NPanelAPI HostPanel)
	{
		HostPanel.setMap(HostPanel.getMap().removeAndUpdate(this));
		//HostSurface.setObjectMap(HostSurface.getObjectMap().addAndUpdate(this));
		
		ArrayList<NPanelRepaintSpace> queue = new ArrayList<NPanelRepaintSpace>();
		int frameDelta = HostPanel.getCurrentFrameDelta();
		if(isDying) 
		{
			animationCounter-=frameDelta;
			needsRepaint=true;
		}	
		//System.out.println("counter: "+animationCounter); 
		boolean redefine = true;
		if(PropertyContainer instanceof NCircleMenu) 
		{
			if(((NCircleMenu)PropertyContainer).isDying) 
			{
				redefine = false;
				needsRepaint=true;
				isDying=true;
			}
		}
		Object something = ((NPanel)HostPanel).findAnything(X, Y, true, this);
		if(something != PropertyContainer &&  redefine==true) 
		{
			animationCounter =-2*CounterLimit;
			PropertyContainer = something;
			mapping = null;
			checked = null;
			needsRepaint=true;
		}
		//-------------------------------------------------------
		boolean check = false;
		//-----------------------------------------------
		if(PropertyContainer instanceof NCircleMenu) 
		{
			System.out.println("is menu");
			NCircleMenu menu = (NCircleMenu)PropertyContainer;
			int chosen = menu.testForChosen(X, Y); 
			if(chosen!=baseMenuChoice) 
			{
				check=true;
			}
			if(animationCounter<CounterLimit) 
			{
				double[] adjustment = menu.getButtonCenterOfChoice(menu.testForChoice(X, Y));
				if(adjustment!=null) 
				{ 
					double shiftX = (adjustment[0]-X)*((double)frameDelta/100000000);
					double shiftY = (adjustment[1]-Y)*((double)frameDelta/100000000);
					((NPanel)HostPanel).translatePanel(-(shiftX*HostPanel.getScale()),-(shiftY*HostPanel.getScale()));
					X += shiftX;
					Y += shiftY;
				}
				
			}
			if(menu.intrinsicScale()/birthScale < 1.25 && menu.intrinsicScale()/birthScale > 0.75) 
			{
				((NPanel)HostPanel).scaleAtReal(X, Y, (double)(1-(1000/(double)frameDelta)));
			      birthScale*=(double)(1-(40000/(double)frameDelta));
			}
			 
		}//-----------------------------------------------
		else if(PropertyContainer instanceof NPanelNode) 
		{
			System.out.println("is Unit");
			if(animationCounter<CounterLimit) {
				double shiftX = (((NPanelNode)PropertyContainer).getX()-X)*((double)frameDelta/100000000);
				double shiftY = (((NPanelNode)PropertyContainer).getY()-Y)*((double)frameDelta/100000000);
				((NPanel)HostPanel).translatePanel(-(shiftX*HostPanel.getScale()),-(shiftY*HostPanel.getScale()));
				X += shiftX;
				Y += shiftY;
				double nodeScale = 80/((NPanelNode)PropertyContainer).getUnitRadius();
				if(nodeScale>birthScale*1.1) {birthScale*=1.1;}
				else if(nodeScale<birthScale*0.9) {birthScale*=0.9;}
				}
		}
		if(mapping==null || check==true) 
		{
			if(PropertyContainer instanceof NPanel)
			{
				byte[] newMapping = {-1, 2, 9, 1, 6, 5,4,7};
				boolean[] newChecked = {true, true, true, false, false, false, true, false};
				mapping = newMapping;
				checked = newChecked;
				needsRepaint=true;
			}
			else if(PropertyContainer instanceof NPanelNode)
			{
				byte[] newMapping = {-1,  4, 8};
				boolean[] newChecked = {false, false, false};
				mapping = newMapping;
				checked = newChecked;
				needsRepaint=true;
			}
			else if(PropertyContainer instanceof NVertex)
			{
				
			}
			else if(PropertyContainer instanceof NCircleMenu)
			{//-----------------------------------------------------------------------
				NCircleMenu menu = (NCircleMenu)PropertyContainer;
				int chosen = menu.testForChosen(X, Y);
				System.out.println("chosen of menu: "+chosen);
				baseMenuChoice = chosen;
				//====================================================
				if(chosen <= 0) 
				{
					mapping=null;
					checked=null;
				}
				//====================================================
				else if(chosen==1) 
				{
					animationCounter=-2*CounterLimit;
					byte[]    newMapping = {-1,  5, 6};
					boolean[] newChecked = {true, true, true,};
					mapping = newMapping;
					checked = newChecked;
					needsRepaint=true;
				}
				//====================================================
				else if(chosen==2) 
				{
					animationCounter=-2*CounterLimit;
					byte[]    newMapping = {-1,  2, 3};
					boolean[] newChecked = {true, true, true,};
					mapping = newMapping;
					checked = newChecked;
					needsRepaint=true;
				}
				//====================================================
				else if(chosen==3) 
				{
					
				}
				//====================================================
				else if(chosen==4) //Settings Icon (panel settings)
				{
					if(menu.getMenuStackBaseContainer() instanceof NPanelNode) 
					{
						animationCounter=-2*CounterLimit;
						byte[]    newMapping = {-1,  5, 9};
						boolean[] newChecked = {false, false, false,};
						mapping = newMapping;
						checked = newChecked;
						needsRepaint=true;
						Actor = 
						(int choice, Object Container)->
						{
							double[] center = this.getButtonCenterOfChoice(choice);
							if(choice==5 || choice==9)
							{
								NCircleMenu newButtonMenu 
									= new NCircleMenu(center[0], center[1], birthScale/2, Builder);
								
								newButtonMenu.animationCounter=0;
								Builder.addMenu(newButtonMenu);
							}  
						};
					}
					//panel base
				}
				//====================================================
				else if(chosen==5) //inner node setting!
				{
					animationCounter=-2*CounterLimit;
					byte[]    newMapping = {-1,  7, 6};//Node functions -> custom....
					boolean[] newChecked = {true, true};
					mapping = newMapping;
					checked = newChecked;
					needsRepaint=true;
					Actor = (int choice, Object Container)->{
						if(Container instanceof NPanel) {
						    NPanel panel = (NPanel)Container;
						    //Setting node accordingly
						}
					};
				}
				else if(chosen==9) //Node type menu
				{
					animationCounter=-2*CounterLimit;
					byte[]    newMapping = {-1,  1, 2, 3};//NodeTypes
					boolean[] newChecked = {false, false, false, false};
					mapping = newMapping;
					checked = newChecked;
					needsRepaint=true;
					Actor = (int choice, Object Container)->{
						if(Container instanceof NPanelNode) 
						{//1. turn into default node; 2. turn into super root; 3. turn into basic root;
						    NPanelNode node = (NPanelNode)Container;
						    NVertex neuron = node.getCore();
						    //Setting node accordingly
						    if(choice==1) {NVertex.State.turnIntoTrainableNeuron(neuron);System.out.println("Turning into default node");}
						    if(choice==2) {NVertex.State.turnIntoParentRoot(neuron);System.out.println("Turning into super root node");}
						    //if(choice==3) {NCore.State.turnIntoBasicRoot(neuron);System.out.println("Turning into basic root node");}
						}
						else if(Container instanceof NPanel) {
							 NVertex neuron = this.Builder.getBlueprintNeuron();
							 if(choice==1) {NVertex.State.turnIntoTrainableNeuron(neuron);System.out.println("Turning into default node");}
							 if(choice==2) {NVertex.State.turnIntoParentRoot(neuron);System.out.println("Turning into super root node");}
							 //if(choice==3) {NCore.State.turnIntoBasicRoot(neuron);System.out.println("Turning into basic root node");}
						}
					};
				}
				//====================================================
			}//-----------------------------------------------------------------------
			highlightID=0;
		}
		double radius=ReferenceRadius/birthScale;
		double innerRadius=InnerReferenceRadius/birthScale;
		double primMod= ((double)animationCounter/CounterLimit);
		if(primMod<=0) {primMod=0.0;}

		//ArrayList<NPanelRepaintSpace> queue = null;
		if(animationCounter<CounterLimit && isDying==false) 
		{animationCounter+=frameDelta; needsRepaint=true;}
		if(needsRepaint==true) 
		{
		       double scaledRadius = primMod*radius*1.075;
		       if(scaledRadius<innerRadius) {scaledRadius = innerRadius;}
		       queue.add(new NPanelRepaintSpace(X, Y, (int)scaledRadius, (int)scaledRadius));//  queue.e_add(new NPanelRepaintSpace(X, Y, (int)radius, (int)radius));
		       //if(X!=oldX || Y!=oldY) {queue.e_add(new NPanelRepaintSpace(oldX, oldY, (int)scaledRadius, (int)scaledRadius)); oldX=X; oldY=Y;}
		}
		else 
		{animationCounter=CounterLimit;}
		needsRepaint = false; 
		HostPanel.setMap(HostPanel.getMap().addAndUpdate(this));
		return queue;
	}
	public NPanelRepaintSpace getRepaintSpace() 
	{
		double radius=ReferenceRadius/birthScale;
		double primMod= ((double)animationCounter/CounterLimit);
		if(primMod<=0) {primMod=0.0;}
		double scaledRadius = primMod*radius*0.65+radius*0.375;
		return new NPanelRepaintSpace(X, Y, (int)scaledRadius, (int)scaledRadius);
	}
	//----------------------------------------------------------
	private Object getMenuStackBaseContainer() 
	{
		if(PropertyContainer instanceof NCircleMenu) {return ((NCircleMenu)PropertyContainer).getMenuStackBaseContainer();}
		return PropertyContainer;
	}
	//----------------------------------------------------------
	private void drawMenuIcon(byte IconID, double x, double y,Graphics2D brush) 
	{
		switch(IconID) 
		{
			case 0:
				break;
			case -1:Icon_CloseMenu(x,y,brush);
			break;
			case 1:Icon_DefaultNode(x,y,brush);
			break;
			case 2:Icon_SuperRoot(x, y, brush);//Icon_SuperRoot(x,y,brush);
			break;
			case 3:Icon_BasicRoot(x,y,brush);
			break;
			case 4:Icon_PanelSettings(x,y,brush);   
			break;
			case 5:Icon_NodeSettings(x,y,brush);   
			break;	
			case 6:Icon_TanhNode(x,y,brush);
			break;
			case 7:Icon_NodeFunctionSettings(x,y,brush);
			break;
			case 8:Icon_BackEndNodeWindow(x, y, brush);	
			break;
			case 9:Icon_NodeType(x,y,brush);
			break;
			default:
			break;
		}
	}
	private void clickedAtChoice(int IconID) {
		NSound sound = new NSound("button_click.wav","./sound/menu/");
		sound.start();
		Actor.applyAction(IconID, getMenuStackBaseContainer());
		System.out.println("IconID: "+IconID);
		switch(IconID) 
		{
			case 0:
			break;
			case -1:click_CloseMenu();
			break;
			case 1:click_DefaultNode();
			break;
			case 2:click_SuperRoot();
			break;
			case 3:click_BasicRoot();
			break;
			case 4:click_PanelSettings();   
			break;
			case 5:click_NodeSettings();   
			break;	
			case 6:click_TanhNode();
			break;
			case 7:click_NodeFunctionSettings();
			break;
			case 8:click_BackEndNodeWindow();	
			break;
			default:
			break;
		}
	}
	//----------------------------------------------------------
	@Override
	public void movementAt(double x, double y, NPanelAPI HostSurface) {
		if(mapping==null) {return;}
		int test = testForChoice(x,y);
		if(test!=highlightID) {
			NSound sound = new NSound("button_hover.wav","./sound/menu/");
			sound.start();
			needsRepaint=true; System.out.println("works");}
		highlightID = test;
	}
	//-------------------------------------------------------------------------------
	private int testForChoice(double x, double y) 
	{
		if(mapping==null) {return 0;}
		double radius=ReferenceRadius/birthScale;
		double innerRadius=InnerReferenceRadius/birthScale;
		double alpha = 2*Math.PI/mapping.length;
		double rotationX=Math.cos(rotation-alpha)*0-Math.sin(rotation-alpha)*-1;
		double rotationY=Math.sin(rotation-alpha)*0+Math.cos(rotation-alpha)*-1;
		
		for(int i=0; i<mapping.length; i++)
		{
			double newRotationX = Math.cos(alpha)*rotationX-Math.sin(alpha)*rotationY;
			double newRotationY = Math.sin(alpha)*rotationX+Math.cos(alpha)*rotationY;
			rotationX = newRotationX;
			rotationY = newRotationY; 
			if((Math.abs(rotationX) + Math.abs(rotationY)) != 1) 
		    {
				double correction = 2-Math.pow(Math.pow(rotationX, 2) + Math.pow(rotationY, 2), 0.5);
				rotationX *= correction;
				rotationY *= correction;
		    }
			double ovalRadius = (animationCounter/CounterLimit)*(radius-innerRadius)/2;
			double ovalX =(X)+rotationX*(ovalRadius+innerRadius)*((double)animationCounter/CounterLimit);
			double ovalY =(Y)+rotationY*(ovalRadius+innerRadius)*((double)animationCounter/CounterLimit);
			double distance = Math.pow(Math.pow(x-ovalX, 2)+Math.pow(y-ovalY, 2), 0.5);
			if(distance<ovalRadius) {return i+1;}
		}
	 return 0;
	}
	//-------------------------------------------------------------------------------
	public int testForChosen(double x, double y) 
	{
		int choice = testForChoice(x,y);
		//System.out.println("testForChose: choice: "+choice);
		if(choice > 0 & mapping!=null) {return mapping[choice-1];}
		return choice;
	}
	//-------------------------------------------------------------------------------
	@Override
	public boolean clickedAt(double x, double y, NPanelAPI HostPanel) 
	{int choice = testForChoice(x,y); 
	 System.out.println("clickedAt: choice: "+choice);
	 if(choice!=0) {clickedAtChoice(mapping[choice-1]); return true;}return false;}
	
	@Override
	public boolean doubleClickedAt(double x, double y, NPanelAPI HostPanel) 
	{
		return false;
	}
	
	//-------------------------------------------------------------------------------
	public boolean testForBody(double x, double y){
		if(testForCenter(x,y)) {return true;}
		if(testForChoice(x,y)!=0) {return true;}
		return false;
	}
	//-------------------------------------------------------------------------------
	private boolean testForCenter(double x, double y)
	{
		double innerRadius=InnerReferenceRadius/birthScale;
		double distance = Math.pow(Math.pow(X-x, 2)+Math.pow(Y-y, 2), 0.5);
		if(distance<innerRadius) {return true;}
		return false;
	}
	
	//------------------------------------------------
	interface MenuActor{public abstract void applyAction(int ActionID, Object Property);}
	//-------------------------------------------------------------------------------
	@Override
	public ArrayList<NPanelRepaintSpace> moveDirectional(double[] data, NPanelAPI Surface)// double startX, double startY, double targX, double targY
	{
		double startX=data[0];
		double startY=data[1]; 
		double targX=data[2]; 
		double targY=data[3];
		ArrayList<NPanelRepaintSpace> queue = new ArrayList<NPanelRepaintSpace>();
		if(testForChoice(startX, startY)!=0 && testForCenter(startX,startY)==false) {
			Surface.setMap(Surface.getMap().removeAndUpdate(this));;
		    queue.add(getRepaintSpace());
			rotation += Math.atan2((targY-Y),(targX-X)) - Math.atan2((startY-Y),(startX-X));
			Surface.setMap(Surface.getMap().addAndUpdate(this));;
		}
		else 
		{	
			Surface.setMap(Surface.getMap().removeAndUpdate(this));;
			queue.add(getRepaintSpace());
			X-=startX-targX; Y-=startY-targY;
			queue.add(getRepaintSpace());
			Surface.setMap(Surface.getMap().addAndUpdate(this));;
		}
        needsRepaint=true;
		return queue;
	}
	
	@Override
	public ArrayList<NPanelRepaintSpace> moveCircular(double[] data, NPanelAPI Surface) //double centerX, double centerY, double x, double y
	{
		return null;
	}
 
	@Override
	public ArrayList<NPanelRepaintSpace> moveTo(double[] data, NPanelAPI Surface) {//double x, double y
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	public double getLeftPeripheral() {return this.X-this.getRadius();}
	@Override
	public double getTopPeripheral() {return this.Y-this.getRadius();}
	@Override
	public double getRightPeripheral() {return this.X+this.getRadius();}
	@Override
	public double getBottomPeripheral() {return this.Y+this.getRadius();}
	@Override
	public int getLayerID() {
		//if(PropertyContainer instanceof NPanelCircleMenu) {
		//	NPanelCircleMenu menu = (NPanelCircleMenu)PropertyContainer;
		//	if(menu!=this) {
		//		return menu.getLayerID()+1;
		//	}
		//}
		return layerID;
	}
	@Override
	public boolean needsRepaintOnLayer(int layerID) {if(this.getLayerID()==layerID) {return true;}return false;}
	@Override
	public void repaintLayer(int layerID, Graphics2D brush, NPanelAPI HostSurface) 
	{
		//HostSurface.setObjectMap(HostSurface.getObjectMap().removeAndUpdate(this));
		this.drawMenu(brush);
		//HostSurface.setObjectMap(HostSurface.getObjectMap().addAndUpdate(this));
	}
	@Override
	public double getRadius() {return ReferenceRadius/birthScale;}



	//=================================================================================================================================================
	//--------------------------------
	private void Icon_CloseMenu(double x, double y, Graphics2D brush) {
		double radius=(ReferenceRadius*(animationCounter/CounterLimit))/birthScale;
		double innerRadius=(InnerReferenceRadius*(animationCounter/CounterLimit))/birthScale;
		double ovalRadius = (radius-innerRadius)/3;
		brush.setColor(new Color(255, 70, 50));
		brush.fillOval((int)(x-ovalRadius), (int)(y-ovalRadius), (int)(ovalRadius*2), (int)(ovalRadius*2));
		ovalRadius*=0.75;
		brush.rotate(Math.PI*0.25, x, y);
		brush.setColor(Color.BLACK);
		brush.fillRect((int)(x-ovalRadius), (int)(y-ovalRadius/5), (int)(ovalRadius*2), (int)(ovalRadius*2/5));
		brush.fillRect((int)(x-ovalRadius/5), (int)(y-ovalRadius), (int)(ovalRadius*2/5), (int)(ovalRadius*2));
		brush.rotate(-Math.PI*0.25, x, y);
	}
	private void click_CloseMenu() {isDying=true;  System.out.println("Klicked at close");}
	//------------------------------------------------
	private void Icon_BasicRoot(double x, double y, Graphics2D brush) {
		double radius=(ReferenceRadius*(animationCounter/CounterLimit))/birthScale;
		double innerRadius=(InnerReferenceRadius*(animationCounter/CounterLimit))/birthScale;
		double ovalRadius = (radius-innerRadius)/5;
		brush.setColor(Color.BLACK);
		double oX = x+ovalRadius/2.55;
		double oY = y+ovalRadius/2.55;
		
		Icon_NodeBase(x, y, brush);
		brush.setColor(Color.CYAN);
		brush.fillRect((int)(oX-ovalRadius/2), (int)(oY-ovalRadius/16),
			       (int)(2*ovalRadius/2), (int)(2*ovalRadius/16));
		brush.fillRect((int)(oX-ovalRadius/16), (int)(oY-ovalRadius/2),
			       (int)(2*ovalRadius/16), (int)(2*ovalRadius/2));
		brush.rotate(Math.PI/4,oX, oY);
		brush.fillRect((int)(oX-ovalRadius/2), (int)(oY-ovalRadius/16),
			       (int)(2*ovalRadius/2), (int)(2*ovalRadius/16));
		brush.fillRect((int)(oX-ovalRadius/16), (int)(oY-ovalRadius/2),
			       (int)(2*ovalRadius/16), (int)(2*ovalRadius/2));
		brush.rotate(-Math.PI/4,oX, oY );
		
	}
	private void click_BasicRoot() {
		
	}
	//------------------------------------------------
	private void Icon_NodeType(double x, double y, Graphics2D brush) {
		//Icon_DefaultNode(x,y,brush);
		Icon_NodeBase(x, y, brush);
		brush.setColor(Color.cyan);
		double radius=(ReferenceRadius*(animationCounter/CounterLimit))/birthScale;
		double innerRadius=(InnerReferenceRadius*(animationCounter/CounterLimit))/birthScale;
		double ovalRadius = ((radius-innerRadius)/5);
		double oX = x+ovalRadius/2.55;
		double oY = y+ovalRadius/2.55;
		Font txt = new Font("Verdana", Font.LAYOUT_LEFT_TO_RIGHT, (int)((70/birthScale)*(animationCounter/CounterLimit)));
		brush.setFont(txt);char[] c = new char[1]; c[0] = '?';
		brush.drawChars(c, 0, 1, (int)(oX-ovalRadius/2), (int)(oY+ovalRadius/1.5));
	}
	//------------------------------------------------
	private void Icon_DefaultNode(double x, double y, Graphics2D brush) {
		double radius=(ReferenceRadius*(animationCounter/CounterLimit))/birthScale;
		double innerRadius=(InnerReferenceRadius*(animationCounter/CounterLimit))/birthScale;
		double ovalRadius = ((radius-innerRadius)/5);
		if(ovalRadius<=0) {return;}
		brush.setColor(Color.BLACK);
		double oX = x+ovalRadius/2.55;
		double oY = y+ovalRadius/2.55;
		brush.fillOval((int)(oX-ovalRadius), (int)(oY-ovalRadius),
				       (int)(2*ovalRadius), (int)(2*ovalRadius));
		brush.setColor(neuronBlue);
		//brush.setColor(Color.BLACK);
		brush.fillOval((int)(oX-ovalRadius*0.8), (int)(oY-ovalRadius*0.8),
			       (int)(2*ovalRadius*0.8), (int)(2*ovalRadius*0.8));
		//----------------------------
		brush.setColor(Color.BLACK);
		ovalRadius = (radius-innerRadius)/15;
		oX = (x-ovalRadius);
		oY = (y-ovalRadius);
		brush.setStroke(new BasicStroke((int) (ovalRadius/2)));
		brush.drawLine((int)(oX), (int)(oY), (int)(oX-3*ovalRadius), (int)(oY));
		brush.rotate(Math.PI/4, oX, oY);
		brush.drawLine((int)(oX), (int)(oY), (int)(oX-3*ovalRadius), (int)(oY));
		brush.rotate(Math.PI/4, oX, oY);
		brush.drawLine((int)(oX), (int)(oY), (int)(oX-3*ovalRadius), (int)(oY));
		brush.rotate(-Math.PI/2, oX, oY);
		
		brush.fillOval((int)(oX-ovalRadius), (int)(oY-ovalRadius),
			       (int)(2*ovalRadius), (int)(2*ovalRadius));
		brush.setColor(Color.CYAN);
		brush.fillRect((int)(oX-ovalRadius/2), (int)(oY-ovalRadius/6),
			       (int)(2*ovalRadius/2), (int)(2*ovalRadius/6));
		brush.fillRect((int)(oX-ovalRadius/6), (int)(oY-ovalRadius/2),
			       (int)(2*ovalRadius/6), (int)(2*ovalRadius/2));
		
		
	}
	private void click_DefaultNode() {
		
	}
	//------------------------------------------------
	private void Icon_NodeFunctionSettings(double x, double y, Graphics2D brush) {
		
		Icon_DefaultNode(x,y,brush);
		double radius=(ReferenceRadius*(animationCounter/CounterLimit))/birthScale;
		double innerRadius=(InnerReferenceRadius*(animationCounter/CounterLimit))/birthScale;
		double ovalRadius = (radius-innerRadius)/5;
		brush.setColor(Color.BLACK);
		double oX = x+ovalRadius/2.55;
		double oY = y+ovalRadius/2.55;
		double scaleHolder = birthScale;
		birthScale *= 2.5;
		Icon_PanelSettings(oX, oY, brush);
		birthScale = scaleHolder;
	}
	private void click_NodeFunctionSettings() {
		
	}
	//------------------------------------------------
		private void Icon_SuperRoot(double x, double y, Graphics2D brush) {

			double radius=(ReferenceRadius*(animationCounter/CounterLimit))/birthScale;
			double innerRadius=(InnerReferenceRadius*(animationCounter/CounterLimit))/birthScale;
			
			double ovalRadius = (radius-innerRadius)/3;
			
			brush.setColor(Color.BLACK);
			double oX = x-ovalRadius/14;
			double oY = y-ovalRadius/14;
			brush.fillOval((int)(oX-ovalRadius), (int)(oY-ovalRadius),
					       (int)(2*ovalRadius), (int)(2*ovalRadius));
			
			brush.setColor(SystemOcean);
			brush.fillOval((int)(oX-ovalRadius*0.9), (int)(oY-ovalRadius*0.9),
				       (int)(2*ovalRadius*0.9), (int)(2*ovalRadius*0.9));
			brush.setClip(null);

			ovalRadius = (radius-innerRadius)/5;
			brush.setColor(Color.BLACK);
			oX = x+ovalRadius/2;
			oY = y+ovalRadius/2;
			double scaleHolder = birthScale;
			birthScale *= 1.25;
			Icon_DefaultNode(oX, oY, brush);
			birthScale = scaleHolder;
		
		}
		private void click_SuperRoot() {
			
		}
		//------------------------------------------------
		private void Icon_PanelSettings(double x, double y, Graphics2D brush) {
					
			double radius=(ReferenceRadius*(animationCounter/CounterLimit))/birthScale;
			double innerRadius=(InnerReferenceRadius*(animationCounter/CounterLimit))/birthScale;
			brush.setColor(Color.BLACK);
			Icon_CogwheelBase(x, y, brush);				
			double ovalRadius = (radius-innerRadius)/4;
			brush.setColor(Color.BLACK);
			brush.fillOval((int)(x-ovalRadius*0.5), (int)(y-ovalRadius*0.5),
						   (int)(2*ovalRadius*0.5), (int)(2*ovalRadius*0.5));

				}
				private void click_PanelSettings() {
					
				}
				//------------------------------------------------
				
				
		private void Icon_NodeSettings(double x, double y, Graphics2D brush) {
					
			//double radius=(R*(animationCounter/CounterLimit))/birthScale;
			//double innerRadius=(r*(animationCounter/CounterLimit))/birthScale;
					
			brush.setColor(Color.BLACK);	
			Icon_CogwheelBase(x, y, brush);
						
			double scaleHolder = birthScale;
			birthScale *= 1.575;
			Icon_DefaultNode(x, y, brush);
			birthScale = scaleHolder;

		}
		private void click_NodeSettings() {
							
		}			
				
				
		private void Icon_BackEndNodeWindow(double x, double y, Graphics2D brush){
			
			//double radius=(R*(animationCounter/CounterLimit))/birthScale;
			double innerRadius=(InnerReferenceRadius*(animationCounter/CounterLimit))/birthScale;
			
			brush.setColor(Color.BLACK);
			
			double a = innerRadius*1.1;
			double b = innerRadius*1.0;
			brush.fillRect((int)(x-a/2), (int)(y-b/2), (int)a, (int)b);
			
			brush.setColor(SystemOcean);
			brush.fillRect((int)(x+a*0.025), (int)(y-b/3.1), (int)(a/2.2-a*0.05), (int)(b/2.33));
			brush.fillRect((int)(x-a/2.2+a*0.025), (int)(y-b/2.3), (int)(a/2.2-a*0.05), (int)(b/3.55));
			double scaleHolder = birthScale;
			birthScale *= 5;
			brush.setColor(Color.BLACK);
			this.Icon_SigmoidBase((((x+a/4.375))), (((y-b/9))), brush);
			birthScale = scaleHolder;
			brush.setColor(Color.CYAN);
			brush.fillRect((int)(((x+a*0.025))), (int)(((y-b/2.3))), (int)((a/2.2)-a*0.05), (int)(b/15));
			
			brush.fillRect((int)(((x+a*0.025))), (int)(((y+b*0.15))), (int)((a/2.2-a*0.05)), (int)(b/15));
			brush.fillRect((int)(((x-a/2.2+a*0.025))), (int)(((y-b*0.1))), (int)((a/2.2)-a*0.05), (int)(b/15));
			Area clipArea = new Area(new Rectangle.Double((x-a/2), (y-b/2), a, b));
			brush.setClip(clipArea);
			brush.setColor(SystemOcean);
			scaleHolder = birthScale;
			birthScale *= 3.25;
			this.Icon_CogwheelBase(x-a/4.25, y+b/4.5, brush);
			birthScale = scaleHolder;
			brush.setClip(null);
		}		
				
		private void click_BackEndNodeWindow() {
			
			if(PropertyContainer instanceof NPanelNode) {
				
				((NPanelNode)PropertyContainer).
					getCore().
						addModule(
								new NVControlFrame("Neuron "+((NPanelNode)PropertyContainer).getCore().getID(),((NPanelNode)PropertyContainer).getCore()));
			}
			
		}
	//===============================================================================
	//===============================================================================
				//Basic subcomponents:
				private void Icon_NodeBase(double x, double y, Graphics2D brush) {
					
					double radius=(ReferenceRadius*(animationCounter/CounterLimit))/birthScale;
					double innerRadius=(InnerReferenceRadius*(animationCounter/CounterLimit))/birthScale;
					double ovalRadius = (radius-innerRadius)/5;
					if(ovalRadius<=0) {return;}
					brush.setColor(Color.BLACK);
					double oX = x+ovalRadius/2.55;
					double oY = y+ovalRadius/2.55;
					brush.fillOval((int)(oX-ovalRadius), (int)(oY-ovalRadius),
							       (int)(2*ovalRadius), (int)(2*ovalRadius));

				//----------------------------
					brush.setColor(Color.BLACK);
					ovalRadius = (radius-innerRadius)/15;
					oX = (x-ovalRadius);
					oY = (y-ovalRadius);
					brush.setStroke(new BasicStroke((int) (ovalRadius/2)));
					brush.drawLine((int)(oX), (int)(oY), (int)(oX-3*ovalRadius), (int)(oY));
					brush.rotate(Math.PI/4, oX, oY);
					brush.drawLine((int)(oX), (int)(oY), (int)(oX-3*ovalRadius), (int)(oY));
					brush.rotate(Math.PI/4, oX, oY);
					brush.drawLine((int)(oX), (int)(oY), (int)(oX-3*ovalRadius), (int)(oY));
					brush.rotate(-Math.PI/2, oX, oY);
					
					brush.fillOval((int)(oX-ovalRadius), (int)(oY-ovalRadius),
						           (int)(2*ovalRadius),  (int)(2*ovalRadius));
					
					brush.setColor(Color.CYAN);
					brush.fillRect((int)(oX-ovalRadius/2), (int)(oY-ovalRadius/6),
						           (int)(2*ovalRadius/2),  (int)(2*ovalRadius/6));
					
					brush.fillRect((int)(oX-ovalRadius/6), (int)(oY-ovalRadius/2),
						           (int)(2*ovalRadius/6),  (int)(2*ovalRadius/2));

				}
				private void Icon_CogwheelBase(double x, double y, Graphics2D brush) {
					
					double radius=
							(ReferenceRadius*(animationCounter/CounterLimit))/birthScale;
					double innerRadius=
							(InnerReferenceRadius*(animationCounter/CounterLimit))/birthScale;
					
					double ovalRadius = 
							((radius-innerRadius)/4);
					
					//brush.setColor(Color.BLACK);
					double oX = x;
					double oY = y;
					
					//brush.fillRect((int)(x-ovalRadius), (int)(y-ovalRadius), (int)(2*ovalRadius), (int)(2*ovalRadius));
					
					Area clipArea = new Area(new Rectangle.Double((x-radius), (y-radius), (2*radius), (2*radius)));
					clipArea.subtract(new Area(new Ellipse2D.Double((oX-ovalRadius*0.7), (oY-ovalRadius*0.7),(2*ovalRadius*0.7), (2*ovalRadius*0.7))));
					brush.setClip(clipArea);
					
					Polygon p = new Polygon(); 
					p.addPoint((int)(oX-ovalRadius/4), (int)(oY+ovalRadius/10));
					p.addPoint((int)(oX-ovalRadius/7), (int)(oY-ovalRadius/3));
					p.addPoint((int)(oX+ovalRadius/7), (int)(oY-ovalRadius/3));
					p.addPoint((int)(oX+ovalRadius/4), (int)(oY+ovalRadius/10));
					p.translate(0, (int)(-ovalRadius));
					int max= 9;
					for(int i=0; i<max; i++) {
						brush.rotate((2*Math.PI/max), oX, oY);
					    brush.fillPolygon(p);
					}
					brush.fillOval((int)(oX-ovalRadius), (int)(oY-ovalRadius),
							       (int)(2*ovalRadius), (int)(2*ovalRadius));

					brush.setClip(null);
					
				}
			
			private void Icon_SigmoidBase(double x, double y, Graphics2D brush) 
			{
				
				double radius=(ReferenceRadius*(animationCounter/CounterLimit))/birthScale;
				double innerRadius=(InnerReferenceRadius*(animationCounter/CounterLimit))/birthScale;

				Area clipArea;
				double r = (radius-innerRadius)/2.5;
				brush.fillRect((int)(x-r*1.25), (int)(y-1.25*r/9),
					       (int)(2*r*1.25), (int)(2*1.25*r/9));
				brush.fillRect((int)(x-1.25*r/9), (int)(y-r*1.25),
					       (int)(2*1.25*r/9), (int)(2*r*1.25));
				
				double cX = x-r*0.2;
				double cY = y-r+r*0.1;
				clipArea = new Area(new Rectangle.Double((x), (y-r), (r), (r)));
				clipArea.subtract(new Area(new Ellipse2D.Double((cX+r*0.375), (cY+0.175*r), (2*r+0.4*r), (2*r-0.35*r))));
				brush.setClip(clipArea);
				//brush.setColor(Color.BLUE);
				brush.fillOval((int)(cX), (int)(cY), (int)(2*r+r*0.4), (int)(2*r));
				//brush.rotate(-Math.PI, x, y);
				cX = x-2*r-r*0.2;
				cY = y-r-r*0.1;
				clipArea = new Area(new Rectangle.Double((x-r), (y), (r), (r)));
				clipArea.subtract(new Area(new Ellipse2D.Double((cX-r*0.4), (cY+0.2*r), (2*r+r*0.4), (2*r-0.4*r))));
				brush.setClip(clipArea);
				brush.fillOval((int)(cX), (int)(cY), (int)(2*r+r*0.4), (int)(2*r));
				brush.setClip(null);
			}
			//------------------------------------------------x
			private void Icon_TanhNode(double x, double y, Graphics2D brush) {
				double radius=(ReferenceRadius*(animationCounter/CounterLimit))/birthScale;
				double innerRadius=(InnerReferenceRadius*(animationCounter/CounterLimit))/birthScale;
				double ovalRadius = (radius-innerRadius)/5;
				brush.setColor(Color.BLACK);
				double oX = x+ovalRadius/2.55;
				double oY = y+ovalRadius/2.55;
				
				this.Icon_DefaultNode(x, y, brush);
				
				double scaleHolder = birthScale;
				birthScale *= (4.5);
				brush.setColor(Color.BLACK);
				this.Icon_SigmoidBase(oX, oY, brush);
				birthScale = scaleHolder;

			}
			private void click_TanhNode() {
				
			}
			@Override
			public boolean hasGripAt(double x, double y, NPanelAPI HostPanel) 
			{
				return this.testForBody(x, y);
			}
			

		

}









