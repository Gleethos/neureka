
package neureka.gui.swing;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.MultipleGradientPaint.CycleMethod;
import java.awt.Paint;
import java.awt.RadialGradientPaint;
import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.Random;


public class SurfaceNodeInput implements SurfaceObject{

    private boolean _isActive = false;
    private boolean _changeOccurred = false;

    private double _radius;

    private double _X;
    private double _Y;

    private double _velX;
    private double _velY;

    public void addToVel(double velX, double velY) {
        _velX += velX;
        _velY += velY;
    }

    private static final int defaultUnitDiameter = 1680;

    SurfaceNodeInput(double neuronX, double neuronY, double unitRadius) {//referenceRadius = 0.1;
        Random dice = new Random();
        double alpha = dice.nextDouble();
        double mod = this.getInputNodeOrbitModFrom(unitRadius);
        _radius = getInputNodeRadiusFrom(unitRadius)*1.75;
        _X = neuronX + Math.cos(alpha) * unitRadius * mod;
        _Y = neuronY + Math.sin(alpha) * unitRadius * mod;
    }

    private double getInputNodeRadiusFrom(double unitRadius) {
        return 0.25 * unitRadius * (Math.pow((1.5 - unitRadius / defaultUnitDiameter), 3));
    }

    private double getInputNodeOrbitModFrom(double unitRadius) {
        return 1.15 + (1 * (0.375 - unitRadius / defaultUnitDiameter));
    }
    //===================================================================================================

    public double angleOf(double x, double y) {
        double l = lengthOf(_X - x, y - _Y);
        return -Math.atan2((_Y - y) / l, (_X - x) / l);
    }

    private double lengthOf(double vecX, double vecY) {
        return Math.pow(Math.pow(vecX, 2) + Math.pow(vecY, 2), 0.5);
    }

    @Override
    public SurfaceRepaintSpace getRepaintSpace() {
        double radius = 1.1 * _radius;//getInputNodeRadiusFrom(unitRadius);
        return new SurfaceRepaintSpace(
                _X-radius, _Y-radius, _X+radius, _Y+radius
                );
    }

    public void paint(Graphics2D brush) {
        //System.out.println(brush.getColor());
        double radius = _radius;//getInputNodeRadiusFrom(neuronRadius);
        //brush.setColor(Color.GREEN);
        brush.fillOval((int) (_X - radius), (int) (_Y - radius), (int) radius * 2, (int) radius * 2);
        Point2D center = new Point2D.Float((int) _X, (int) _Y);
        float gradientRadius = (float) radius;
        float[] dist = {0.0f, 0.7f, 0.8f};//0, 7, 8
        Color[] colors = {Color.BLACK, Color.BLACK, brush.getColor()};
        Paint oldPaint = brush.getPaint();
        RadialGradientPaint rgp
                = new RadialGradientPaint(center, gradientRadius, dist, colors, CycleMethod.NO_CYCLE);
        brush.setPaint(rgp);
        brush.fillOval((int) (_X - radius), (int) (_Y - radius), (int) radius * 2, (int) radius * 2);
        brush.setPaint(oldPaint);

    }

    public void paintTop(Graphics2D brush, String display, double transparancy) {
        double radius = _radius;//getInputNodeRadiusFrom(neuronRadius);
        brush.setColor(Color.BLACK);
        brush.fillOval((int) (_X - (radius * 0.72)), (int) (_Y - (radius * 0.72)), (int) (radius * 0.72) * 2, (int) (radius * 0.72) * 2);
        if (transparancy > 0.01 && display.length() > 0) {
            if (transparancy > 1) {
                transparancy = 1;
            }
            brush.setColor(
                    new Color(
                            (int) (Color.CYAN.getRed() * transparancy),
                            (int) (Color.CYAN.getGreen() * transparancy),
                            (int) (Color.CYAN.getBlue() * transparancy))
            );
            switch (display) {
                case "+":
                    DrawUtils.IconPlus(_X, _Y, _radius * 0.85, brush);
                    break;
                case "*":
                    DrawUtils.IconStar(_X, _Y, _radius * 0.85, brush);
                    break;
                case "tanh":
                    DrawUtils.IconSigmoidBase(_X, _Y, brush, 1, radius*0.00525);
                    break;
                default:
                    Font ValueFont = new Font("SansSerif", Font.BOLD, ((int)(2 * radius) / (display.length())));
                    brush.setFont(ValueFont);
                    brush.drawString(
                            display,
                            (int) (_X - (display.length() * ValueFont.getSize() * 0.3)),
                            (int) (_Y + (ValueFont.getSize() * 0.45))
                    );
            }

        }
    }

    //===================================================================================================
    public boolean isActive() {
        return _isActive;
    }

    public void setIsActive(boolean active) {
        _isActive = active;
    }

    @Override
    public void moveCircular(double[] data, Surface Surface) {
        if (data.length == 3) {
           moveCircularBy(data, Surface);
           return;
        }
        double centerX = data[0];
        double centerY = data[1];
        double x = data[2];
        double y = data[3];
        double vecX = _X - centerX;
        double vecY = _Y - centerY;
        double newVecX = x - centerX;
        double newVecY = y - centerY;
        double alpha = -1 * (Math.atan2(vecX, vecY) - Math.atan2(newVecY, newVecX));

        //ArrayList<SurfaceRepaintSpace> queue = new ArrayList<SurfaceRepaintSpace>();
        //queue.add(
        //    Surface.layers()[7].add(getRepaintSpace(Surface));
        //);
        vecX = Math.cos(alpha) * vecX - Math.sin(alpha) * vecY;
        vecY = Math.cos(alpha) * vecY + Math.sin(alpha) * vecX;
        _X = centerX + vecX;
        _Y = centerY + vecY;
        //if (queue.get(0).X == _X && queue.get(0).Y == _Y) {
        //    queue.remove(0);
        //    return queue;
        //}
        _changeOccurred = true;
        //queue.add(this.getRepaintSpace());
        //Surface.layers()[7].add(getRepaintSpace(Surface));
        //return queue;
    }

    //
    private void moveCircularBy(double[] data, Surface Surface) {
        //ArrayList<SurfaceRepaintSpace> queue = new ArrayList<SurfaceRepaintSpace>();
        if (data[0] < 1e-3) return;// queue;
        double alpha = data[0];
        double centerX = data[1];
        double centerY = data[2];
        double vecX = _X - centerX;
        double vecY = _Y - centerY;
        vecX = Math.cos(alpha) * vecX - Math.sin(alpha) * vecY;
        vecY = Math.cos(alpha) * vecY + Math.sin(alpha) * vecX;
        _X = centerX + vecX;
        _Y = centerY + vecY;
        //if (queue.get(0).X == _X && queue.get(0).Y == _Y) {
        //    queue.remove(0);
        //    return queue;
        //}
        _changeOccurred = true;
        //queue.add(
        //        Surface.layers()[7].add(
        //        this.getRepaintSpace(Surface)
        //    );
        return;// queue;
    }

    public void updateOn(double centerX, double centerY, double hostRadius, Surface surface) {
        _X += _velX;
        _Y += _velY;
        _velX *= 0.8;
        _velY *= 0.8;

        hostRadius *= this.getInputNodeOrbitModFrom(hostRadius);
        double vecX = _X - centerX;
        double vecY = _Y - centerY;
        double l = lengthOf(vecX, vecY);
        _X = centerX + (vecX / l) * hostRadius;
        _Y = centerY + (vecY / l) * hostRadius;
        _changeOccurred = true;
    }


    public boolean changeOccured() {
        return _changeOccurred;
    }

    public void forgetChange() {
        _changeOccurred = false;
    }

    //===================================================================================================
    public boolean testFor(double x, double y) {
        double d = Math.pow((Math.pow((x - _X), 2) + Math.pow((y - _Y), 2)), 0.5);
        double radius = _radius;//getInputNodeRadiusFrom(unitRadius);
        if (d <= radius) {
            return true;
        }
        return false;
    }
    //==========================================================================================================

    public void moveDirectional(double[] data, Surface Surface) {
        double startX = data[0];
        double startY = data[1];
        double targX = data[2];
        double targY = data[3];
        //ArrayList<SurfaceRepaintSpace> queue = new ArrayList<SurfaceRepaintSpace>();
        //queue.add(getRepaintSpace(Surface));
        //Surface.layers()[7].add(getRepaintSpace(Surface));
        //queue.getFrom(0).dY+=100;
        double vecX = targX - startX;
        double vecY = targY - startY;
        _X += vecX;
        _Y += vecY;
        //queue.add(getRepaintSpace(Surface));
        //Surface.layers()[7].add(getRepaintSpace(Surface));
        //return queue;
    }

    @Override
    public double getX() {
        return _X;
    }

    @Override
    public double getY() {
        return _Y;
    }

    @Override
    public boolean killable() {
        return false;
    }

    @Override
    public boolean hasGripAt(double x, double y, Surface HostPanel) {
        return false;
    }

    @Override
    public void moveTo(double[] data, Surface Surface) {

    }

    @Override
    public void updateOn(Surface HostPanel) {

    }

    @Override
    public void movementAt(double x, double y, Surface HostPanel) {

    }

    @Override
    public boolean clickedAt(double x, double y, Surface HostPanel) {
        return false;
    }

    @Override
    public boolean doubleClickedAt(double x, double y, Surface HostPanel) {
        return false;
    }

    @Override
    public double getRadius() {
        return _radius;
    }

    @Override
    public double getLeftPeripheral() {
        return _X-_radius;
    }

    @Override
    public double getTopPeripheral() {
        return _Y+_radius;
    }

    @Override
    public double getRightPeripheral() {
        return _X+_radius;
    }

    @Override
    public double getBottomPeripheral() {
        return _Y-_radius;
    }



    @Override
    public boolean needsRepaintOnLayer(int layerID) {
        return false;
    }

    @Override
    public void repaintLayer(int layerID, Graphics2D brush, Surface HostSurface) {

    }

    @Override
    public int getLayerID() {
        return 0;
    }



}
