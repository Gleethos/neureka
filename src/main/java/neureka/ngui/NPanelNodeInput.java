
package neureka.ngui;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.MultipleGradientPaint.CycleMethod;
import java.awt.Paint;
import java.awt.RadialGradientPaint;
import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.Random;


public class NPanelNodeInput implements NPanelObject{
	
	private boolean isActive = false;
	private boolean changeOccured=false;

	private double X;
	private double Y;
	
	private static final int defaultUnitDiameter = 1680;

	NPanelNodeInput(double neuronX, double neuronY, double unitRadius){//referenceRadius = 0.1;
	Random dice = new Random();
	double alpha = dice.nextDouble();
	double mod = this.getInputNodeOrbitModFrom(unitRadius);
	X = neuronX + Math.cos(alpha)*unitRadius*mod;
	Y = neuronY + Math.sin(alpha)*unitRadius*mod;
	}
	private double getInputNodeRadiusFrom(double unitRadius) {return 0.25*unitRadius*(Math.pow((1.5-unitRadius/defaultUnitDiameter),3));}
	private double getInputNodeOrbitModFrom(double unitRadius) {return 1.15+(1*(0.375-unitRadius/defaultUnitDiameter));}
	//===================================================================================================
	
	public double angleOf(double x, double y) {double l = lengthOf(X-x, y-Y);return -Math.atan2((Y-y)/l, (X-x)/l);}
	
	private double lengthOf(double vecX, double vecY) {return Math.pow(Math.pow(vecX, 2)+Math.pow(vecY, 2), 0.5);}
	
	public NPanelRepaintSpace getRepaintSpace(double unitRadius) {
		double radius = 1.1*getInputNodeRadiusFrom(unitRadius);
		return new NPanelRepaintSpace(X, Y, radius, radius);
	}
	
	public void paint(Graphics2D brush, double unitX, double unitY , double neuronRadius) 
	{
		//System.out.println(brush.getColor());
		double radius = getInputNodeRadiusFrom(neuronRadius);
		//brush.setColor(Color.GREEN);
		brush.fillOval((int)(X-radius), (int)(Y-radius), (int)radius*2, (int)radius*2);
		Point2D center = new Point2D.Float((int)X, (int)Y);
		float gradientRadius =  (float)radius;
		float[] dist = { 0.0f, 0.7f, 0.8f };//0, 7, 8
		Color[] colors = { Color.BLACK, Color.BLACK, brush.getColor() }; 
		Paint oldPaint = brush.getPaint();
		RadialGradientPaint rgp 
		= new RadialGradientPaint(center, gradientRadius, dist, colors, CycleMethod.NO_CYCLE);
		brush.setPaint(rgp);
	    brush.fillOval((int)(X-radius), (int)(Y-radius), (int)radius*2, (int)radius*2);
		brush.setPaint(oldPaint);
	
	}
	public void paintTop(Graphics2D brush, double neuronX, double neuronY , double neuronRadius, String display, double transparancy) 
	{
		System.out.println(transparancy);
		double radius = getInputNodeRadiusFrom(neuronRadius);
		brush.setColor(Color.BLACK);
		brush.fillOval((int)(X-(radius*0.72)), (int)(Y-(radius*0.72)), (int)(radius*0.72)*2, (int)(radius*0.72)*2);
		if(transparancy>0.01) 
		{ 
			if(transparancy>1) {transparancy=1;}
			brush.setColor(new Color(
			(int)(Color.CYAN.getRed()*transparancy), 
			(int)(Color.CYAN.getGreen()*transparancy), 
			(int)(Color.CYAN.getBlue()*transparancy)));
			brush.fillRect((int)(X-(radius*0.55)), (int)(Y-(radius*0.1)), (int)(radius*0.55)*2, (int)(radius*0.1)*2);
			brush.fillRect((int)(X-(radius*0.1)),  (int)(Y-(radius*0.55)),(int)(radius*0.1)*2, (int)(radius*0.55)*2);
			int col = (int)(0);
			brush.setColor(new Color(col,col,col));
			Font ValueFont = new Font("SansSerif", Font.BOLD, ((int)(2*radius)/10));
			brush.setFont(ValueFont);
			brush.drawString(display, (int)(X-(display.length()*ValueFont.getSize()*0.33)), (int)(Y+(ValueFont.getSize()*0.38)));
		}
	}
	//===================================================================================================
	public boolean isActive() {return isActive;}
	public void setIsActive(boolean active) {isActive = active;}
	@Override
	public ArrayList<NPanelRepaintSpace> moveCircular(double[] data, NPanel_I Surface) {
		//return moveCircularAndGetRepaintQueue(angleBetween(X-centerX, Y-centerY,x-centerX, y-centerY), centerX, centerY);
		if(data.length==3) {return moveCircularBy(data, Surface);}
		double centerX = data[0]; 
		double centerY = data[1]; 
		double x = data[2]; 
		double y = data[3];
		double vecX = X-centerX;
		double vecY = Y-centerY;
		double newVecX = x-centerX;
		double newVecY = y-centerY;
		double alpha = -1*(Math.atan2(vecX, vecY)-Math.atan2(newVecY, newVecX));
		
		double unitRadius = Math.pow((Math.pow(vecX, 2)+Math.pow(vecY, 2)), 0.5);
		ArrayList<NPanelRepaintSpace> queue = new ArrayList<NPanelRepaintSpace>();
		queue.add(this.getRepaintSpace(unitRadius));
		vecX = Math.cos(alpha)*vecX-Math.sin(alpha)*vecY;
		vecY = Math.cos(alpha)*vecY+Math.sin(alpha)*vecX;		    
		X = centerX + vecX;
		Y = centerY + vecY;
		if(queue.get(0).X==X&&queue.get(0).Y==Y) {queue.remove(0); return queue;}
		changeOccured=true;
		queue.add(this.getRepaintSpace(unitRadius));
		return queue;
	}
	//@Override
	private ArrayList<NPanelRepaintSpace> moveCircularBy(double[] data, NPanel_I Surface) {
		//alpha=2*Math.PI-alpha;
		//alpha+=Math.PI/2;
		double alpha = data[0]; 
		double centerX = data[1]; 
		double centerY = data[2];
		double unitRadius = Math.pow((Math.pow(X-centerX, 2)+Math.pow(Y-centerY, 2)), 0.5);
		ArrayList<NPanelRepaintSpace> queue = new ArrayList<NPanelRepaintSpace>();
		queue.add(this.getRepaintSpace(unitRadius));
		double vecX = X-centerX;
		double vecY = Y-centerY;
		vecX = Math.cos(alpha)*vecX-Math.sin(alpha)*vecY;
		vecY = Math.cos(alpha)*vecY+Math.sin(alpha)*vecX;	
		X = centerX + vecX;
		Y = centerY + vecY;
		if(queue.get(0).X==X&&queue.get(0).Y==Y) {queue.remove(0); return queue;}
		changeOccured=true;
		queue.add(this.getRepaintSpace(1.5*unitRadius));
		return queue;
	}

	public ArrayList<NPanelRepaintSpace> adjustOrbit(double centerX, double centerY, double unitRadius) {
		ArrayList<NPanelRepaintSpace> queue = new ArrayList<NPanelRepaintSpace>();
		queue.add(this.getRepaintSpace(unitRadius));
		unitRadius *= this.getInputNodeOrbitModFrom(unitRadius);
		double vecX = X-centerX;
		double vecY = Y-centerY;
		double l = lengthOf(vecX, vecY);
		X = centerX+(vecX/l)*unitRadius;
		Y = centerY+(vecY/l)*unitRadius;
		if(queue.get(0).X==X&&queue.get(0).Y==Y) {queue.remove(0); return queue;}
		changeOccured=true;
		queue.add(this.getRepaintSpace(unitRadius));
		return queue;
	}
	
	public ArrayList<NPanelRepaintSpace> adjustWithRespectTo(double newVecX, double newVecY, double unitX, double unitY, ArrayList<NPanelNodeInput> nodeList) {
		ArrayList<NPanelRepaintSpace> queue = new ArrayList<NPanelRepaintSpace>();
		double vecX = X-unitX;
		double vecY = Y-unitY;
		double alpha = -1*(Math.atan2(vecX, vecY)-Math.atan2(newVecY, newVecX));	
		alpha*=0.5;
		
		for(int Ii=0; Ii<nodeList.size(); Ii++) {
			if(nodeList.get(Ii)!=this) {
				double delta = -(nodeList.get(Ii).angleOf(unitX, unitY)-this.angleOf(unitX, unitY));
				System.out.println("[Ii:"+Ii+"]->Delta: "+delta);
				if(Math.abs(delta)<Math.PI/4) {
					delta = Math.PI/4-delta;
					System.out.println("[Ii:"+Ii+"]->rotating!");
					nodeList.get(Ii).noticeChange();
					this.noticeChange();
					if(alpha*delta<0) {delta*=-1;}
					double[] data = {delta, unitX, unitY};
					queue.addAll(nodeList.get(Ii).moveCircular(data, null));
					
				}
			}
		}
		double[] data = {alpha, unitX, unitY};
		queue.addAll(this.moveCircular(data, null));
		return queue;
	}
	
	public boolean changeOccured() {return changeOccured;}
	public void noticeChange() {changeOccured = true;}
	public void forgetChange() {changeOccured = false;}
	//===================================================================================================
	public boolean testFor(double x, double y, double unitRadius) 
	{
		double d = Math.pow((Math.pow((x-X), 2) + Math.pow((y-Y), 2)), 0.5);
		double radius = getInputNodeRadiusFrom(unitRadius);
		if(d<=radius) 
		{return true;} 
		return false;
	}
	//==========================================================================================================
	@Override
	public ArrayList<NPanelRepaintSpace> moveDirectional(double[] data, NPanel_I Surface)
	{
		
		double startX = data[0];
		double startY = data[1]; 
		double targX = data[2]; 
		double targY = data[3];
		ArrayList<NPanelRepaintSpace> queue = new ArrayList<NPanelRepaintSpace>();
		queue.add(getRepaintSpace(defaultUnitDiameter/2));
		//queue.getFrom(0).dY+=100;
		double vecX = targX-startX;
		double vecY = targY-startY;
		X+=vecX;
		Y+=vecY;
		queue.add(getRepaintSpace(defaultUnitDiameter/2));
		return queue;
	}
	@Override
	public ArrayList<NPanelRepaintSpace> moveTo(double[] data, NPanel_I Surface)
	{
		double x = data[0]; 
		double y = data[1];
		ArrayList<NPanelRepaintSpace> queue = new ArrayList<NPanelRepaintSpace>();
		queue.add(this.getRepaintSpace(defaultUnitDiameter/2));
		///queue.getFrom(0).dX+=100;
		X=x;
		Y=y;
		queue.add(this.getRepaintSpace(defaultUnitDiameter/2));
		return queue;
	}
	@Override
	public ArrayList<NPanelRepaintSpace> updateOn(NPanel_I HostPanel) {
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	public double getX() {
		// TODO Auto-generated method stub
		return X;
	}
	@Override
	public double getY() {
		// TODO Auto-generated method stub
		return Y;
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
	public boolean needsRepaintOnLayer(int layerID) {
		// TODO Auto-generated method stub
		return false;
	}
	@Override
	public void repaintLayer(int layerID, Graphics2D brush, NPanel_I HostSurface) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public int getLayerID() {
		// TODO Auto-generated method stub
		return 0;
	}
	@Override
	public boolean killable() {
		// TODO Auto-generated method stub
		return false;
	}
	@Override
	public boolean hasGripAt(double x, double y, NPanel_I HostPanel) {
		// TODO Auto-generated method stub
		return false;
	}
	@Override
	public void movementAt(double x, double y, NPanel_I HostPanel) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public double getRadius() {
		// TODO Auto-generated method stub
		return 0;
	}
	@Override
	public boolean clickedAt(double x, double y, NPanel_I HostPanel) {
		// TODO Auto-generated method stub
		return false;
	}
	@Override
	public boolean doubleClickedAt(double x, double y, NPanel_I HostPanel) {
		// TODO Auto-generated method stub
		return false;
	}
	@Override
	public NPanelRepaintSpace getRepaintSpace() {
		// TODO Auto-generated method stub
		return null;
	}
	
}
