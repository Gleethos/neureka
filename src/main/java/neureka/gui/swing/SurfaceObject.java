package neureka.gui.swing;

import java.awt.Graphics2D;
import java.util.ArrayList;

public interface SurfaceObject {
	
	public boolean killable();
	public boolean hasGripAt(double x, double y, Surface HostPanel);
	
	//public ArrayList<SurfaceRepaintSpace> moveCircular(Movement context);//double centerX, double centerY, double x, double y
	public ArrayList<SurfaceRepaintSpace> moveCircular(double[] data, Surface Surface);//double alpha, double centerX, double centerY
	public ArrayList<SurfaceRepaintSpace> moveDirectional(double[] data, Surface Surface);//double startX, double startY, double targX, double targY
	public ArrayList<SurfaceRepaintSpace> moveTo(double[] data, Surface Surface);//double x, double y
	public ArrayList<SurfaceRepaintSpace> updateOn(Surface HostPanel);
	
	public abstract void movementAt(double x, double y, Surface HostPanel);
	public abstract boolean clickedAt(double x, double y, Surface HostPanel);
	public abstract boolean doubleClickedAt(double x, double y, Surface HostPanel);
	
	public abstract double getX();
	public abstract double getY();
	
	public abstract double getRadius();
	
	public abstract double getLeftPeripheral();
	public abstract double getTopPeripheral();
	public abstract double getRightPeripheral();
	public abstract double getBottomPeripheral();
	
	
	public SurfaceRepaintSpace getRepaintSpace();
	
	//public abstract int getLayerID();
	public abstract boolean needsRepaintOnLayer(int layerID);
	public abstract void repaintLayer(int layerID, Graphics2D brush, Surface HostSurface);
	public int getLayerID();
}
