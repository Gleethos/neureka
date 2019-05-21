package neureka.ngui;

public interface NPanelAPI
{

	//Interaction:
	//--------------------------------------------------------------------------------------------------------------------------------
		public void doubleClickedAt(int x, int y);
	//--------------------------------------------------------------------------------------------------------------------------------
		public void clickedAt(int x, int y);
	//--------------------------------------------------------------------------------------------------------------------------------	
		public void pressedAt(int x, int y);
	//--------------------------------------------------------------------------------------------------------------------------------	
		public void longpressedAt(int x, int y);
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

		public NAnimator getAnimator();
		public NMapAPI   getMap();
		
		public void setMap(NMapAPI newMap);
		
		public void releasedAt(int x, int y);
	
}
