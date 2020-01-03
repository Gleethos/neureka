package neureka.gui.swing;

import java.awt.*;
import java.util.List;

public interface Surface
{

	public interface ObjectPainter{
		void paint(Graphics2D brush);
	}
	//Interaction:
	//--------------------------------------------------------------------------------------------------------------------------------
		public void doubleClickedAt(int x, int y);
	//--------------------------------------------------------------------------------------------------------------------------------
		public void clickedAt(int x, int y);
	//--------------------------------------------------------------------------------------------------------------------------------	
		public void pressedAt(int x, int y);
	//--------------------------------------------------------------------------------------------------------------------------------	
		public void longPressedAt(int x, int y);
	//--------------------------------------------------------------------------------------------------------------------------------	
		public void draggedAt(int x, int y);
	//--------------------------------------------------------------------------------------------------------------------------------
		public void draggedBy(int[] Vector);
	//--------------------------------------------------------------------------------------------------------------------------------	
		public void movementAt(int x, int y);
	//--------------------------------------------------------------------------------------------------------------------------------	  
		public void scaleAt(int x, int y, double scale);
	//--------------------------------------------------------------------------------------------------------------------------------	

		public int getHeight();
		public int getWidth();
		
		public double realX(double x);
		public double realY(double y);
		
		public int getCurrentFrameDelta();
		public int getFrameDelta();
		
		public double getScale();

		public Animator getAnimator();
		public AbstractSpaceMap getMap();
		
		public void setMap(AbstractSpaceMap newMap);
		
		public void releasedAt(int x, int y);

		public List<ObjectPainter>[] layers();

		public SurfaceRepaintSpace getCurrentFrameSpace();

}
