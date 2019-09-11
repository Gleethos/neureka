package neureka.ngui.swing;

import java.awt.Graphics2D;
import java.util.ArrayList;

public interface NPanelObject {
	
	public boolean killable();
	public boolean hasGripAt(double x, double y, NPanel_I HostPanel);
	
	//public ArrayList<NPanelRepaintSpace> moveCircular(Movement context);//double centerX, double centerY, double x, double y
	public ArrayList<NPanelRepaintSpace> moveCircular(double[] data, NPanel_I Surface);//double alpha, double centerX, double centerY
	public ArrayList<NPanelRepaintSpace> moveDirectional(double[] data, NPanel_I Surface);//double startX, double startY, double targX, double targY
	public ArrayList<NPanelRepaintSpace> moveTo(double[] data, NPanel_I Surface);//double x, double y
	public ArrayList<NPanelRepaintSpace> updateOn(NPanel_I HostPanel);
	
	public abstract void movementAt(double x, double y, NPanel_I HostPanel);
	public abstract boolean clickedAt(double x, double y, NPanel_I HostPanel);
	public abstract boolean doubleClickedAt(double x, double y, NPanel_I HostPanel);
	
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
	public abstract void repaintLayer(int layerID, Graphics2D brush, NPanel_I HostSurface);
	public int getLayerID();
}
