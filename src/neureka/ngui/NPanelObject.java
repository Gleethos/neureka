package neureka.ngui;

import java.awt.Graphics2D;
import java.util.ArrayList;

public interface NPanelObject {
	
	public boolean killable();
	public boolean hasGripAt(double x, double y, NPanelAPI HostPanel);
	
	//public ArrayList<NPanelRepaintSpace> moveCircular(Movement context);//double centerX, double centerY, double x, double y
	public ArrayList<NPanelRepaintSpace> moveCircular(double[] data, NPanelAPI Surface);//double alpha, double centerX, double centerY
	public ArrayList<NPanelRepaintSpace> moveDirectional(double[] data, NPanelAPI Surface);//double startX, double startY, double targX, double targY
	public ArrayList<NPanelRepaintSpace> moveTo(double[] data, NPanelAPI Surface);//double x, double y
	public ArrayList<NPanelRepaintSpace> updateOn(NPanelAPI HostPanel);
	
	public abstract void movementAt(double x, double y, NPanelAPI HostPanel);
	public abstract boolean clickedAt(double x, double y, NPanelAPI HostPanel);
	public abstract boolean doubleClickedAt(double x, double y, NPanelAPI HostPanel);
	
	public abstract double getX();
	public abstract double getY();
	
	public abstract double getRadius();
	
	public abstract double getLeftPeripheral();
	public abstract double getTopPeripheral();
	public abstract double getRightPeripheral();
	public abstract double getBottomPeripheral();
	
	
	public NPanelRepaintSpace getRepaintSpace();
	
	//public abstract int getLayerID();
	public abstract boolean needsRepaintOnLayer(int layerID);
	public abstract void repaintLayer(int layerID, Graphics2D brush, NPanelAPI HostSurface);
	public int getLayerID();
}
