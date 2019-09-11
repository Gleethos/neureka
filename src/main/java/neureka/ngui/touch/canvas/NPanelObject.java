package neureka.ngui.touch.canvas;

import javafx.scene.canvas.GraphicsContext;

import java.util.ArrayList;

public interface NPanelObject {
	
	public boolean killable();
	public boolean hasGripAt(double x, double y, TSurface_I HostPanel);
	
	//public ArrayList<NPanelRepaintSpace> moveCircular(Movement context);//double centerX, double centerY, double x, double y
	public ArrayList<NPanelRepaintSpace> moveCircular(double[] data, TSurface_I Surface);//double alpha, double centerX, double centerY
	public ArrayList<NPanelRepaintSpace> moveDirectional(double[] data, TSurface_I Surface);//double startX, double startY, double targX, double targY
	public ArrayList<NPanelRepaintSpace> moveTo(double[] data, TSurface_I Surface);//double x, double y
	public ArrayList<NPanelRepaintSpace> updateOn(TSurface_I HostPanel);
	
	public abstract void movementAt(double x, double y, TSurface_I HostPanel);
	public abstract boolean clickedAt(double x, double y, TSurface_I HostPanel);
	public abstract boolean doubleClickedAt(double x, double y, TSurface_I HostPanel);
	
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
	public abstract void repaintLayer(Layer layer, GraphicsContext brush, TSurface_I HostSurface);
	public Layer getLayer();
}
