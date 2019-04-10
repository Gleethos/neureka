
package ngui;

public class NPanelRepaintSpace {

	public double X, Y;
	public double dX, dY;
	
	NPanelRepaintSpace(double centerX, double centerY, double distanceX, double distanceY)
	{X=centerX;Y=centerY;dX=distanceX;dY=distanceY;}
	
	public double getCenterX(){return X;}
	public double getCenterY(){return Y;}
	public double getDistanceX(){return dX;}
	public double getDistanceY(){return dY;}
	
	public void include(double x, double y) 
	{
		double LTCornerX =  X-dX;
		double LTCornerY = Y-dY;
		
		double RBCornerX = X+dX;
		double RBCornerY = Y+dY;
	
		if(x<X-dX)      {LTCornerX=x;}
		else if(x>X+dX) {RBCornerX=x;}
		
		if(y<Y-dY)      {RBCornerY=y;}
		else if(y>Y+dY) {LTCornerY=y;}
		
		X=(LTCornerX+RBCornerX)/2;
		Y=(LTCornerY+RBCornerY)/2;
		
		dX=X-LTCornerX;
		dY=Y-RBCornerY;
	}
	
	
	public void include(double x, double y, double dx, double dy) 
	{
		
		double LTCornerX = X-dX;
		double LTCornerY = Y-dY;
		
		double RBCornerX = X+dX;
		double RBCornerY = Y+dY;
	
		if(Math.abs(x-dx)>Math.abs(X-dX))      {LTCornerX=x-dx;}
		if(Math.abs(x+dx)>Math.abs(X+dX))      {RBCornerX=x+dx;}
		
		if(Math.abs(y-dy)>Math.abs(Y-dY))      {RBCornerY=y-dy;}
		if(Math.abs(y+dy)>Math.abs(Y+dY))      {LTCornerY=y+dy;}
		
		X=(LTCornerX+RBCornerX)/2;
		Y=(LTCornerY+RBCornerY)/2;
		
		dX=Math.abs(X-RBCornerX);
		dY=Math.abs(Y-RBCornerY);
	}
	
	
	
}
