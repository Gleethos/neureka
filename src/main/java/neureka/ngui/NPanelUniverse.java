package neureka.ngui;

import java.awt.*;
import java.util.ArrayList;

public class NPanelUniverse implements NPanelObject
{

	private int X;
	private int Y;
	private int Radius;
	//NPanelMapNode Map;
	
	
	
	
	//@Override
	public ArrayList<NPanelRepaintSpace> moveCircular(double centerX, double centerY, double x, double y) {
		// TODO Auto-generated method stub
		return null;
	}

	//@Override
	public ArrayList<NPanelRepaintSpace> moveCircular(double alpha, double centerX, double centerY) {
		// TODO Auto-generated method stub
		return null;
	}

	//@Override
	public ArrayList<NPanelRepaintSpace> moveDirectional(double startX, double startY, double targX, double targY) {
		// TODO Auto-generated method stub
		return null;
	}

	//@Override
	public ArrayList<NPanelRepaintSpace> moveTo(double x, double y) {
		// TODO Auto-generated method stub
		return null;
	}

	//@Override
	public ArrayList<NPanelRepaintSpace> updateOn(NPanel HostPanel) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean killable() {
		return false;
	}

	@Override
	public boolean hasGripAt(double x, double y, NPanel_I HostPanel) {
		return false;
	}

	@Override
	public ArrayList<NPanelRepaintSpace> moveCircular(double[] data, NPanel_I Surface) {
		return null;
	}

	@Override
	public ArrayList<NPanelRepaintSpace> moveDirectional(double[] data, NPanel_I Surface) {
		return null;
	}

	@Override
	public ArrayList<NPanelRepaintSpace> moveTo(double[] data, NPanel_I Surface) {
		return null;
	}

	@Override
	public ArrayList<NPanelRepaintSpace> updateOn(NPanel_I HostPanel) {
		return null;
	}

	@Override
	public void movementAt(double x, double y, NPanel_I HostPanel) {

	}

	@Override
	public boolean clickedAt(double x, double y, NPanel_I HostPanel) {
		return false;
	}

	@Override
	public boolean doubleClickedAt(double x, double y, NPanel_I HostPanel) {
		return false;
	}

	@Override
	public double getX() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getY() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getRadius() {
		return 0;
	}

	@Override
	public double getLeftPeripheral() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getTopPeripheral() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getRightPeripheral() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getBottomPeripheral() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public NPanelRepaintSpace getRepaintSpace() {
		return null;
	}

	@Override
	public boolean needsRepaintOnLayer(int layerID) {
		return false;
	}

	@Override
	public void repaintLayer(int layerID, Graphics2D brush, NPanel_I HostSurface) {

	}

	@Override
	public int getLayerID() {
		// TODO Auto-generated method stub
		return 0;
	}

}
