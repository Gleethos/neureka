package neureka.ngui.touch.canvas;

import java.awt.event.*;

public class NListener implements MouseListener, MouseMotionListener, MouseWheelListener
{
	int[] SwipeVector = null;
	int[] PressPoint = null;
	
	int clickedX;
	int clickedY;
	
	int currentMouseX;
	int currentMouseY;
	
	private static final int pressTimeLimit = 900500000;
	private static final double pressRadius = 100;

	
	TSurface_I Surface;
	
	NListener(TSurface_I surface)
	{
		Surface = surface;
	}
//--------------------------------------------------------------------------------------------------------------------------------	
	public void updateOn(TSurface gui)
	{
		if(PressPoint!=null) {
			 if(PressPoint[0]>pressTimeLimit) {
				 Surface.longPressedAt(PressPoint[1], PressPoint[2]);
			 }
			 else 
			 {
			 	PressPoint[0]+=gui.getFrameDelta();
			 }
		}
	}
//--------------------------------------------------------------------------------------------------------------------------------	
	public void mouseClicked(MouseEvent e) 
	{  
		if(clickedX==e.getX() && clickedY==e.getY())
		{
			Surface.doubleClickedAt(e.getX(), e.getY());
		}			
		else{Surface.clickedAt(e.getX(), e.getY());collapseDragVector();} 
		clickedX = e.getX();
		clickedY = e.getY();
		currentMouseX = e.getX();
		currentMouseY = e.getY();
	}
//--------------------------------------------------------------------------------------------------------------------------------	
	public void mouseEntered(MouseEvent e) 
	{
		
	}
//--------------------------------------------------------------------------------------------------------------------------------	
	public void mouseExited(MouseEvent e) 
	{
		
	}
//--------------------------------------------------------------------------------------------------------------------------------	
	public void mousePressed(MouseEvent e) 
	{
		Surface.pressedAt(e.getX(), e.getY());
		addDragPoint(e.getX(), e.getY()); 
		collapseDragVector(); 
		addDragPoint(e.getX(), e.getY()); 
		PressPoint = new int[3];
		PressPoint[0]=0;
		PressPoint[1]=e.getX();
		PressPoint[2]=e.getY();
		System.out.println("pressed");
		currentMouseX = e.getX();
		currentMouseY = e.getY();
	}
//--------------------------------------------------------------------------------------------------------------------------------	
	public void mouseReleased(MouseEvent e) 
	{
		collapseDragVector(); 
		Surface.releasedAt(e.getX(), e.getY());
		PressPoint=null; 
		currentMouseX = e.getX();
		currentMouseY = e.getY();
	}
//--------------------------------------------------------------------------------------------------------------------------------	
	public void mouseDragged(MouseEvent e) 
	{
		addDragPoint(e.getX(),e.getY());
		Surface.draggedBy(SwipeVector);
	
		Surface.draggedAt(e.getX(), e.getY());
	
		currentMouseX = e.getX();
		currentMouseY = e.getY();
	}
//--------------------------------------------------------------------------------------------------------------------------------
	public void addDragPoint(int x, int y) 
	{
		if(SwipeVector != null) {
			if(SwipeVector.length == 4) {
				SwipeVector[2]=x;
				SwipeVector[3]=y;	
			} else {
				   int startX = SwipeVector[0];
				   int startY = SwipeVector[1];
				   SwipeVector = new int[4];
				   SwipeVector[0]=startX;
				   SwipeVector[1]=startY;
				   SwipeVector[2]=x;
				   SwipeVector[3]=y;
			   }
		   } else {
			SwipeVector = new int[2];
			SwipeVector[0] = x;
			SwipeVector[1] = y;
		   }
	}
//--------------------------------------------------------------------------------------------------------------------------------
	public void setDragStart(int startX, int startY)
	{
		if(SwipeVector!=null) {
			SwipeVector[0] = startX;
			SwipeVector[1] = startY;
		}
		else {addDragPoint(startX, startY);}
	}
//--------------------------------------------------------------------------------------------------------------------------------
	private void collapseDragVector() 
	{
		if(SwipeVector==null) 
		{
			return;
		}
		if(SwipeVector.length<4 || SwipeVector.length==2) 
		{
			return;
		}
		int[] old = SwipeVector;
		SwipeVector = new int[2];
		SwipeVector[0]=old[2];
		SwipeVector[1]=old[3];
	}
//--------------------------------------------------------------------------------------------------------------------------------	
	public void mouseMoved(MouseEvent e)
	{
		Surface.movementAt(e.getX(),e.getY());
		collapseDragVector();
		addDragPoint(e.getX(), e.getY());
		if(PressPoint!=null) 
		{
			double distance = 
					Math.pow(Math.pow(PressPoint[1]-e.getX(), 2)+Math.pow(PressPoint[2]-e.getY(), 2), 0.5);
		 	if(distance>pressRadius) 
		 	{
			 	PressPoint = null; 
		 	}
		}
		currentMouseX = e.getX();
		currentMouseY = e.getY();
	}   
//--------------------------------------------------------------------------------------------------------------------------------	  
	public void mouseWheelMoved(MouseWheelEvent e) 
	{
		Surface.scaleAt(e.getX(), e.getY(), (1+((double)e.getWheelRotation()/30)));
	}
//--------------------------------------------------------------------------------------------------------------------------------	
}