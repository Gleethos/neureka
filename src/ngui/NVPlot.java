package ngui;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.geom.AffineTransform;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JPanel;

public class NVPlot extends JPanel{
	//-----------------------------------------------------------------------------
		private double graphCenterX=0;
		private double graphCenterY=0;
		
		private NListener Listener;
		private AffineTransform Scaler = new AffineTransform();
		private AffineTransform Translator = new AffineTransform();	

		private int scaleAnimationCounter=50;

		private int grabbedX;
		private int grabbedY;
		private int oldGrabbedX;
		private int oldGrabbedY;
		private boolean grabbed;
		private boolean isDragging;
		
		private boolean needsRepaint = false;
		
		private boolean recentlyClicked=false;	
		private int releasedX;
		private int releasedY;
		private int currentMouseX;
		private int currentMouseY;
		
		private double panelCenterX;
		private double panelCenterY;
		
		private ArrayList<NPanelRepaintSpace> RepaintQueue;

		private double attachedX;
		private double attachedY;
		
		private int[] PressPoint;
		//private int   pressPointTime;
		private static final int pressTimeLimit = 900500000;
		private static final double pressRadius = 100;
		
		private int[] moveVector;

		private boolean touchMode = true;
		
		private boolean drawRepaintSpaces = false;
		private boolean advancedRendering = false;
		private boolean mapRendering = false;
		
		private boolean mouseIsOffscreen = true;
	    	
		private int Shade=0;
		private boolean shadedClipping=false;
		
		private long frameStart;
		private long frameEnd;
		private int frameDelta;
		

	 public NVPlot()
	 {
		 this.setForeground(Color.BLACK);
	     Listener = new NListener();
		 this.addMouseListener(Listener);
		 this.addMouseWheelListener(Listener);
		 this.addMouseMotionListener(Listener);
		 
		 this.setLayout(new BorderLayout());
		 JButton button = new JButton("test button");
		 this.add(button, BorderLayout.SOUTH);
		 
		 repaint(0, 0, getWidth(), getHeight());	
		 //Scaler.scale(0.5,0.5);   
	 }
	 
	 public int getCurrentFrameDelta() {return frameDelta;}

	 public void updateAndRedraw() {
		 if(PressPoint!=null) {
			 //System.out.println("PP NOT null");
			 if(PressPoint[0]>pressTimeLimit) {
				 
			 }else {PressPoint[0]+=frameDelta;System.out.println("PP: "+PressPoint[0]);}//
		 }
		 update();
		 startRepaintQueue();
		 //System.out.println(frameDelta);
		 frameEnd=System.nanoTime();
		 frameDelta = (int) (Math.abs((frameEnd-frameStart)));
		 frameStart=System.nanoTime();//double seconds = (frameDelta);
	 }
	//================================================================================================================================
	 public void switchTouchMode()         {if(touchMode==false)         {touchMode=true;}        else {touchMode=false;}}
	 public void switchMapRendering()      {if(mapRendering==false)      {mapRendering=true;}     else {mapRendering=false;}}
	 public void switchDrawRepaintSpaces() {if(drawRepaintSpaces==false) {drawRepaintSpaces=true;}else {drawRepaintSpaces=false;}}
	 public void switchAdvancedRendering() {if(advancedRendering==false) {advancedRendering=true;}else {advancedRendering=false;}}
	 public void switchShadedClipping()    {if(shadedClipping==false)    {shadedClipping=true;}   else {shadedClipping=false;}}
	//-------------------------------------------------------------------------
	 public  ArrayList<NPanelRepaintSpace> getRepaintQueue(){return this.RepaintQueue;}
	 public void setRepaintQueue(ArrayList<NPanelRepaintSpace> newQueue) {RepaintQueue = newQueue;}
	//--------------------------------------------------------------------------------------------------------------------------------
	 private void update() 
	 {ArrayList<NPanelRepaintSpace> repaintSpace = new ArrayList<NPanelRepaintSpace>();
	 
	  applyMoveVector();
	  if(RepaintQueue==null) {RepaintQueue= new ArrayList<NPanelRepaintSpace>();}
	  if(repaintSpace!=null) {RepaintQueue.addAll(repaintSpace);}

	  }
	 
	//--------------------------------------------------------------------------------------------------------------------------------
	 public void startRepaintQueue()
	 {
		  if(RepaintQueue!=null) {
	      NPanelRepaintSpace repaintSpace;
	      while(RepaintQueue.size()>0) 
		      {//System.out.println("REPAINTING NOW!");
	    	   repaintSpace = RepaintQueue.get(0); RepaintQueue.remove(0);      
	    	   if(repaintSpace!=null) 
	    	      {//System.out.println(RepaintQueue.size());
	    	       repaintAndScaleOnscreenArea
	               (realToOnPanelX(repaintSpace.getCenterX()), realToOnPanelY(repaintSpace.getCenterY()), repaintSpace.getDistanceX(), repaintSpace.getDistanceY());  	   
	    	      }
	          }
	      RepaintQueue=null;
	      }
	 }
	//-------------------------------------------------------------------------
	 public void paintComponent(Graphics g)
	 {super.paintComponent(g);
		 Graphics2D brush = (Graphics2D) g;	 
		 this.setBackground(Color.BLACK);
		 
		 if(advancedRendering) {brush.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);}
		 else {brush.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);}
		 
		 if(shadedClipping) {
			 Shade%=200*Math.PI; Shade= Math.abs(Shade);
			 int color = (int) Math.abs(((Math.pow(Math.sin((Math.PI/200)*Shade), 3)))*200);
			 color=(int) Math.abs(color);
			 if(color < 0) {color*=-1;}
		 
			 brush.setColor(new Color(color, color, color));
			 brush.fillRect(0, 0, getWidth(), getHeight());
			 Shade++;
			 Shade%=200;
		 }
		 
		 if((panelCenterY!=((double)getHeight())/2)||(panelCenterX!=((double)getWidth())/2)) 
		 {
			 this.translatePanel((((double)getWidth())/2-panelCenterX), (((double)getHeight())/2-panelCenterY));
			 panelCenterX = ((double)getWidth())/2;
			 panelCenterY = ((double)getHeight())/2;
		 }

		 brush.transform(Scaler);
		 brush.transform(Translator);
		 
		 brush.setColor(Color.BLUE);
		 brush.fillRect((int)(graphCenterX), (int)(graphCenterY), 10000, 200);
		 
		 brush.setColor(Color.GREEN);
		 brush.fillOval((int)realX(currentMouseX)-5, (int)realY(currentMouseY)-5, 10, 10);

		 //RENDERING VISUALIZATION:
		 if(drawRepaintSpaces) {
			 if(RepaintQueue!=null) {
			    brush.setColor(Color.WHITE);
			    brush.setStroke(new BasicStroke(30f));
		         for(int i=0; i<RepaintQueue.size(); i++)
		             { 
			           brush.drawRect((int)(RepaintQueue.get(i).getCenterX()-RepaintQueue.get(i).getDistanceX()), (int)(RepaintQueue.get(i).getCenterY()-RepaintQueue.get(i).getDistanceY()), 
					                  (int)(RepaintQueue.get(i).getDistanceX())*2, (int)(RepaintQueue.get(i).getDistanceY())*2);
		             }
		         } 	 
		     }
		 //===
	 }
	 
	//-------------------------------------------------------------------------------------------------------------------------------- 
	 public void movementAt(int x, int y)
	 {	isDragging=false;
		    if(PressPoint!=null) {
				 double distance = Math.pow(Math.pow(PressPoint[1]-x, 2)+Math.pow(PressPoint[2]-y, 2), 0.5);
				 if(distance>pressRadius) 
				 {PressPoint = null;}
				 //System.out.println("movement at distance: "+distance);
				}
		    recentlyClicked=false;	

		    currentMouseX = x;
		    currentMouseY = y;	
	 }
	//-------------------------------------------------------------------------------------------------------------------------------- 		
		private void draggedAt(int x, int y)
		{	
			grabbed = true;
			oldGrabbedX=grabbedX;
			oldGrabbedY=grabbedY;
			grabbedX=x;
			grabbedY=y;	
			if(touchMode) {addMovementPoint(x,y);}
			
			if(PressPoint!=null) {
				double distance = Math.pow(Math.pow(oldGrabbedX-x, 2)+Math.pow(oldGrabbedY-y, 2), 0.5)+Math.pow(Math.pow(PressPoint[1]-x, 2)+Math.pow(PressPoint[2]-y, 2), 0.5);
				double mod = 1*(Math.pow(distance, (distance)));
				PressPoint[0] -= 1*(mod);
				if(PressPoint[0]<0) {PressPoint[0]=0;}
				if(distance>pressRadius) {PressPoint = null;}

				System.out.println("dragged at distance: "+distance+"; mod: "+mod);
			}

			translatePanel((grabbedX-oldGrabbedX),(grabbedY-oldGrabbedY));
			isDragging=true;
		}
	//--------------------------------------------------------------------------------------------------------------------------------
		private void addMovementPoint(int x, int y) {
			
			if(moveVector != null) 
			   {
				if(moveVector.length == 4) 
				   {
					   moveVector[2]=x;
					   moveVector[3]=y;	
				   }
				else 
				   {
					   int startX = moveVector[0];
					   int startY = moveVector[1];
					   moveVector = new int[4];
					   moveVector[0]=startX;
					   moveVector[1]=startY;
					   moveVector[2]=x;
					   moveVector[3]=y;
				   }
			   }
			else 
			   {
				   moveVector = new int[2];
				   moveVector[0] = x;
				   moveVector[1] = y;
			   }
		}
		public void setMovementStart(int startX, int startY) {
			if(moveVector!=null) {
				moveVector[0] = startX;
				moveVector[1] = startY;
			}
			else {addMovementPoint(startX, startY);}
		}
		public void collapseMoveVector() {
			if(moveVector==null) {return;}
			if(moveVector.length<4) {return;}
			int[] old = moveVector;
			  moveVector = new int[2];
			  moveVector[0]=old[2];
			  moveVector[1]=old[3];
		}
		
		//Vektor shift on panel
		public void applyMoveVector() {		
			if(moveVector != null) {if(moveVector.length==2) {return;}} else {return;}

		  collapseMoveVector();
		}

	//--------------------------------------------------------------------------------------------------------------------------------	

		private void repaintAndScaleOnscreenArea(double centerX, double centerY, double distanceX, double distanceY)
		{
			int LTCornerX = (int)(centerX-1-(distanceX*Scaler.getScaleX()));
			int LTCornerY = (int)(centerY-1-(distanceY*Scaler.getScaleY()));
			
			int width =  (int)(2+       2*distanceX*Scaler.getScaleX());
			int height = (int)(2+       2*distanceY*Scaler.getScaleY());
				
			if(LTCornerX < 0) {width+=LTCornerX; LTCornerX=0;}
			if(LTCornerY < 0) {height+=LTCornerY; LTCornerY=0;}
			
			if(width < 0) {width=0;}
			if(height < 0) {height=0;}

			if(LTCornerX > getWidth()) {LTCornerX=getWidth();}
			if(LTCornerY > getHeight()) {LTCornerY=getHeight();}
			
			if(width > getWidth()) {width=getWidth();}
			if(height > getHeight()) {height=getHeight();}
			
		 repaint( LTCornerX, LTCornerY,
	      	      width, height);
		
		 }	
	//--------------------------------------------------------------------------------------------------------------------------------	
		public void translatePanel(double translateX, double translateY) {
			Translator.translate(translateX*1/Scaler.getScaleX(), translateY*1/Scaler.getScaleY());	    
			//repaint(0, 0, getWidth(), getHeight());//Repaint spaces?
			if(RepaintQueue != null) {RepaintQueue = null;}
			repaint(0, 0, getWidth(), getHeight());//Repaint spaces?
		}
		//-------------------------------------------------------------------------------------------------------------------------------- 	
		 	private double realX(double x) {return ((x)/Scaler.getScaleX())-Translator.getTranslateX();}//
		 	private double realY(double y) {return ((y)/Scaler.getScaleY())-Translator.getTranslateY();}//
		 	public double realToOnPanelX(double x) {return ((x+Translator.getTranslateX())*Scaler.getScaleX());}//
		 	public double realToOnPanelY(double y) {return ((y+Translator.getTranslateY())*Scaler.getScaleY());}//
	//-------------------------------------------------------------------------
		public void scaleAtReal(double x, double y, double scaleFactor) 
		{scaleAt((int)realToOnPanelX(x), (int)realToOnPanelY(y), scaleFactor);}
		
		public void scaleAt(int x, int y, double scaleFactor) {
			recentlyClicked=false;
		    currentMouseX = x;
		    currentMouseY = y;
		    //scaling at coursor
		    translatePanel(-x, -y);
		    //Scaler.translate((panelCenterX ), (panelCenterY));
		    Scaler.scale((scaleFactor), (scaleFactor));
		    //Scaler.translate(-(panelCenterX ), -(panelCenterY)); 
		  	translatePanel(+x, +y);
			//translatePanel(-(getWidth()/2-x), -(getHeight()/2-y));
		    repaint(0, 0, getWidth(), getHeight());
		}
		
		
		public void clickedAt(int x, int y) {
			//checkPanelScaleCenter(getWidth()/2, getHeight()/2);
		    setMovementStart(x, y);

			repaint(0, 0, getWidth(), getHeight());
		}
		
		public double getScale() {return Scaler.getScaleX();}
		
		public void pressedAt(int x, int y) {
			PressPoint = new int[3];
			PressPoint[0]=0;
			PressPoint[1]=x;
			PressPoint[2]=y;
			System.out.println("pressed");
		}
		public void releasedAt(int x, int y) {
			PressPoint = null;
		}
		
		public int currentMouseX() {return currentMouseX;}
		public int currentMouseY() {return currentMouseY;}
		
		public double currentRealMouseX() {return realX(currentMouseX);}
		public double currentRealMouseY() {return realY(currentMouseY);}
		//Mouse Events:
	//=======================================================================================================	
	private class NListener implements MouseListener, MouseMotionListener, MouseWheelListener
	{//--------------------------------------------------------------------------------------------------------------------------------	
		public void mouseClicked(MouseEvent e) {clickedAt(e.getX(), e.getY());}
	//--------------------------------------------------------------------------------------------------------------------------------	
		public void mouseEntered(MouseEvent e) {mouseIsOffscreen = false;}
	//--------------------------------------------------------------------------------------------------------------------------------	
		public void mouseExited(MouseEvent e) {
			System.out.println("exited");
			mouseIsOffscreen = true;
			grabbed = false;
		}
	//--------------------------------------------------------------------------------------------------------------------------------	
		public void mousePressed(MouseEvent e) {
			grabbed = true;
			grabbedX=e.getX();
			grabbedY=e.getY();
			pressedAt(e.getX(), e.getY());
		}
	//--------------------------------------------------------------------------------------------------------------------------------	
		public void mouseReleased(MouseEvent e) {
			releasedAt(e.getX(), e.getY());
			grabbed = false;
			releasedX=e.getX();
			releasedY=e.getY();	 
		}
	//--------------------------------------------------------------------------------------------------------------------------------	
		public void mouseDragged(MouseEvent e) {draggedAt(e.getX(), e.getY());}
	//--------------------------------------------------------------------------------------------------------------------------------	
		public void mouseMoved(MouseEvent e) {movementAt(e.getX(),e.getY());}   
	//--------------------------------------------------------------------------------------------------------------------------------	  
		public void mouseWheelMoved(MouseWheelEvent e) {
			double scaleIncrement = (1+((double)e.getWheelRotation()/30));
			scaleAt(e.getX(), e.getY(), scaleIncrement);
			}
	//--------------------------------------------------------------------------------------------------------------------------------	
	    }

		
		
		
		
		
		
		
}
