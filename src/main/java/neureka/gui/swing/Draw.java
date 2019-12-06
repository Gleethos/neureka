package neureka.gui.swing;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.geom.Area;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.awt.geom.RoundRectangle2D;

public class Draw {

    private final static int OUTER_REFERENCE_RADIUS = 300;
    private final static int INNER_REFERENCE_RADIUS = 100;

    //=================================================================================================================================================

    public static void Icon_Star(double x, double y, double r, Graphics2D brush)
    {
        RoundRectangle2D rec = new RoundRectangle2D.Double(
                (x - (r / 10)),
                (y - r * 0.650),
                (2* r / 10),
                (1* r * 0.75),
                (r * 0.4), (r * 0.4)
        );
        int n = 5;
        double theta = 2*Math.PI/n;
        for(int i=0; i<n; i++){
            brush.fill(rec);
            brush.rotate(((i==n-1)?-n+1:1)*theta, x, y);

        }
    }

    public static void Icon_Plus(double x, double y, double r, Graphics2D brush){
        Rectangle2D rec = new Rectangle2D.Double(
                (x - (r / 10)),
                (y - r * 0.650),
                (2* r / 10),
                (1* r * 0.75)
        );
        int n = 4;
        double theta = 2*Math.PI/n;
        for(int i=0; i<n; i++){
            brush.fill(rec);
            brush.rotate(((i==n-1)?-n+1:1)*theta, x, y);

        }
    }

    //------------------------------------------------
    public static void Icon_Close(double x, double y, Graphics2D brush, double birthScale, double animScale) {
        double radius = (OUTER_REFERENCE_RADIUS * animScale) / birthScale;
        double innerRadius = (INNER_REFERENCE_RADIUS * animScale) / birthScale;
        double ovalRadius = (radius - innerRadius) / 3;
        brush.setColor(new Color(255, 70, 50));
        brush.fillOval((int) (x - ovalRadius), (int) (y - ovalRadius), (int) (ovalRadius * 2), (int) (ovalRadius * 2));
        ovalRadius *= 0.75;
        brush.rotate(Math.PI * 0.25, x, y);
        brush.setColor(Color.BLACK);
        brush.fillRect((int) (x - ovalRadius), (int) (y - ovalRadius / 5), (int) (ovalRadius * 2), (int) (ovalRadius * 2 / 5));
        brush.fillRect((int) (x - ovalRadius / 5), (int) (y - ovalRadius), (int) (ovalRadius * 2 / 5), (int) (ovalRadius * 2));
        brush.rotate(-Math.PI * 0.25, x, y);
    }

    //------------------------------------------------

    /*
    public static void Icon_BasicRoot(double x, double y, Graphics2D brush, double birthScale, double animScale) {
        double radius = (OUTER_REFERENCE_RADIUS * animScale) / birthScale;
        double innerRadius = (INNER_REFERENCE_RADIUS * animScale) / birthScale;
        double ovalRadius = (radius - innerRadius) / 5;
        brush.setColor(Color.BLACK);
        double oX = x + ovalRadius / 2.55;
        double oY = y + ovalRadius / 2.55;

        Icon_NodeBase(x, y, brush, birthScale, 1.0);
        brush.setColor(Color.CYAN);
        brush.fillRect((int) (oX - ovalRadius / 2), (int) (oY - ovalRadius / 16),
                (int) (2 * ovalRadius / 2), (int) (2 * ovalRadius / 16));
        brush.fillRect((int) (oX - ovalRadius / 16), (int) (oY - ovalRadius / 2),
                (int) (2 * ovalRadius / 16), (int) (2 * ovalRadius / 2));
        brush.rotate(Math.PI / 4, oX, oY);
        brush.fillRect((int) (oX - ovalRadius / 2), (int) (oY - ovalRadius / 16),
                (int) (2 * ovalRadius / 2), (int) (2 * ovalRadius / 16));
        brush.fillRect((int) (oX - ovalRadius / 16), (int) (oY - ovalRadius / 2),
                (int) (2 * ovalRadius / 16), (int) (2 * ovalRadius / 2));
        brush.rotate(-Math.PI / 4, oX, oY);

    }
    */
    /*
    //------------------------------------------------
    public static void Icon_NodeType(double x, double y, Graphics2D brush, double birthScale, double animScale) {
        //Icon_DefaultNode(x,y,brush, birthScale, 1.0);
        Icon_NodeBase(x, y, brush, birthScale, 1.0);
        brush.setColor(Color.cyan);
        double radius = (OUTER_REFERENCE_RADIUS * animScale) / birthScale;
        double innerRadius = (INNER_REFERENCE_RADIUS * animScale) / birthScale;
        double ovalRadius = ((radius - innerRadius) / 5);
        double oX = x + ovalRadius / 2.55;
        double oY = y + ovalRadius / 2.55;
        Font txt = new Font("Verdana", Font.LAYOUT_LEFT_TO_RIGHT, (int) ((70 / birthScale) * animScale));
        brush.setFont(txt);
        char[] c = new char[1];
        c[0] = '?';
        brush.drawChars(c, 0, 1, (int) (oX - ovalRadius / 2), (int) (oY + ovalRadius / 1.5));
    }
    */
    /*
    //------------------------------------------------
    public static void Icon_DefaultNode(double x, double y, Graphics2D brush, double birthScale, double animScale) {
        double radius = (OUTER_REFERENCE_RADIUS * animScale) / birthScale;
        double innerRadius = (INNER_REFERENCE_RADIUS * animScale) / birthScale;
        double ovalRadius = ((radius - innerRadius) / 5);
        if (ovalRadius <= 0) {
            return;
        }
        brush.setColor(Color.BLACK);
        double oX = x + ovalRadius / 2.55;
        double oY = y + ovalRadius / 2.55;
        brush.fillOval((int) (oX - ovalRadius), (int) (oY - ovalRadius),
                (int) (2 * ovalRadius), (int) (2 * ovalRadius));
        brush.setColor(Palette.NEURAL_BLUE);
        //brush.setColor(Color.BLACK);
        brush.fillOval((int) (oX - ovalRadius * 0.8), (int) (oY - ovalRadius * 0.8),
                (int) (2 * ovalRadius * 0.8), (int) (2 * ovalRadius * 0.8));
        //----------------------------
        brush.setColor(Color.BLACK);
        ovalRadius = (radius - innerRadius) / 15;
        oX = (x - ovalRadius);
        oY = (y - ovalRadius);
        brush.setStroke(new BasicStroke((int) (ovalRadius / 2)));
        brush.drawLine((int) (oX), (int) (oY), (int) (oX - 3 * ovalRadius), (int) (oY));
        brush.rotate(Math.PI / 4, oX, oY);
        brush.drawLine((int) (oX), (int) (oY), (int) (oX - 3 * ovalRadius), (int) (oY));
        brush.rotate(Math.PI / 4, oX, oY);
        brush.drawLine((int) (oX), (int) (oY), (int) (oX - 3 * ovalRadius), (int) (oY));
        brush.rotate(-Math.PI / 2, oX, oY);

        brush.fillOval((int) (oX - ovalRadius), (int) (oY - ovalRadius),
                (int) (2 * ovalRadius), (int) (2 * ovalRadius));
        brush.setColor(Color.CYAN);
        brush.fillRect((int) (oX - ovalRadius / 2), (int) (oY - ovalRadius / 6),
                (int) (2 * ovalRadius / 2), (int) (2 * ovalRadius / 6));
        brush.fillRect((int) (oX - ovalRadius / 6), (int) (oY - ovalRadius / 2),
                (int) (2 * ovalRadius / 6), (int) (2 * ovalRadius / 2));


    }
    */
    /*
    //------------------------------------------------
    public static void Icon_NodeFunctionSettings(double x, double y, Graphics2D brush, double birthScale, double animScale) {

        Icon_DefaultNode(x, y, brush, birthScale, 1.0);
        double radius = (OUTER_REFERENCE_RADIUS * animScale) / birthScale;
        double innerRadius = (INNER_REFERENCE_RADIUS * animScale) / birthScale;
        double ovalRadius = (radius - innerRadius) / 5;
        brush.setColor(Color.BLACK);
        double oX = x + ovalRadius / 2.55;
        double oY = y + ovalRadius / 2.55;
        double scaleHolder = birthScale;
        birthScale *= 2.5;
        Icon_PanelSettings(oX, oY, brush, birthScale, 1.0);
        birthScale = scaleHolder;
    }
    */
    /*
    //------------------------------------------------
    public static void Icon_SuperRoot(double x, double y, Graphics2D brush, double birthScale, double animScale) {

        double radius = (OUTER_REFERENCE_RADIUS * animScale) / birthScale;
        double innerRadius = (INNER_REFERENCE_RADIUS * animScale) / birthScale;

        double ovalRadius = (radius - innerRadius) / 3;

        brush.setColor(Color.BLACK);
        double oX = x - ovalRadius / 14;
        double oY = y - ovalRadius / 14;
        brush.fillOval((int) (oX - ovalRadius), (int) (oY - ovalRadius),
                (int) (2 * ovalRadius), (int) (2 * ovalRadius));

        brush.setColor(neureka.gui.swing.Palette.SYSTEM_OCEAN);
        brush.fillOval((int) (oX - ovalRadius * 0.9), (int) (oY - ovalRadius * 0.9),
                (int) (2 * ovalRadius * 0.9), (int) (2 * ovalRadius * 0.9));
        brush.setClip(null);

        ovalRadius = (radius - innerRadius) / 5;
        brush.setColor(Color.BLACK);
        oX = x + ovalRadius / 2;
        oY = y + ovalRadius / 2;
        double scaleHolder = birthScale;
        birthScale *= 1.25;
        Icon_DefaultNode(oX, oY, brush, birthScale, 1.0);
        birthScale = scaleHolder;

    }
    */
    /*
    //------------------------------------------------
    public static void Icon_PanelSettings(double x, double y, Graphics2D brush, double birthScale, double animScale) {

        double radius = (OUTER_REFERENCE_RADIUS * animScale) / birthScale;
        double innerRadius = (INNER_REFERENCE_RADIUS * animScale) / birthScale;
        brush.setColor(Color.BLACK);
        Icon_CogwheelBase(x, y, brush, birthScale, 1.0);
        double ovalRadius = (radius - innerRadius) / 4;
        brush.setColor(Color.BLACK);
        brush.fillOval((int) (x - ovalRadius * 0.5), (int) (y - ovalRadius * 0.5),
                (int) (2 * ovalRadius * 0.5), (int) (2 * ovalRadius * 0.5));

    }
    //------------------------------------------------
    */
    /*

    public static void Icon_NodeSettings(double x, double y, Graphics2D brush, double birthScale, double animScale) {

        brush.setColor(Color.BLACK);
        Icon_CogwheelBase(x, y, brush, birthScale, 1.0);
        //double scaleHolder = birthScale;
        birthScale *= 1.575;
        Icon_DefaultNode(x, y, brush, birthScale, 1.0);
        //birthScale = scaleHolder;

    }

    */
    /*
    public static void Icon_BackEndNodeWindow(double x, double y, Graphics2D brush, double birthScale, double animScale) {

        double innerRadius = (INNER_REFERENCE_RADIUS * animScale) / birthScale;
        
        brush.setColor(Color.BLACK);
        double a = innerRadius * 1.1;
        double b = innerRadius * 1.0;
        brush.fillRect((int) (x - a / 2), (int) (y - b / 2), (int) a, (int) b);

        brush.setColor(Palette.SYSTEM_OCEAN);
        brush.fillRect((int) (x + a * 0.025), (int) (y - b / 3.1), (int) (a / 2.2 - a * 0.05), (int) (b / 2.33));
        brush.fillRect((int) (x - a / 2.2 + a * 0.025), (int) (y - b / 2.3), (int) (a / 2.2 - a * 0.05), (int) (b / 3.55));
        double scaleHolder = birthScale;
        birthScale *= 5;
        brush.setColor(Color.BLACK);
        Icon_SigmoidBase((((x + a / 4.375))), (((y - b / 9))), brush, birthScale, 1.0);
        birthScale = scaleHolder;
        brush.setColor(Color.CYAN);
        brush.fillRect((int) (((x + a * 0.025))), (int) (((y - b / 2.3))), (int) ((a / 2.2) - a * 0.05), (int) (b / 15));

        brush.fillRect((int) (((x + a * 0.025))), (int) (((y + b * 0.15))), (int) ((a / 2.2 - a * 0.05)), (int) (b / 15));
        brush.fillRect((int) (((x - a / 2.2 + a * 0.025))), (int) (((y - b * 0.1))), (int) ((a / 2.2) - a * 0.05), (int) (b / 15));
        Area clipArea = new Area(new Rectangle.Double((x - a / 2), (y - b / 2), a, b));
        brush.setClip(clipArea);
        brush.setColor(Palette.SYSTEM_OCEAN);
        scaleHolder = birthScale;
        birthScale *= 3.25;
        Icon_CogwheelBase(x - a / 4.25, y + b / 4.5, brush, birthScale, 1.0);
        birthScale = scaleHolder;
        brush.setClip(null);
    }
    */
    /*

    //===============================================================================
    //===============================================================================
    //Basic subcomponents:
    public static void Icon_NodeBase(double x, double y, Graphics2D brush, double birthScale, double animScale) {

        double radius = (OUTER_REFERENCE_RADIUS * animScale) / birthScale;
        double innerRadius = (INNER_REFERENCE_RADIUS * animScale) / birthScale;
        double ovalRadius = (radius - innerRadius) / 5;
        if (ovalRadius <= 0) {
            return;
        }
        brush.setColor(Color.BLACK);
        double oX = x + ovalRadius / 2.55;
        double oY = y + ovalRadius / 2.55;
        brush.fillOval((int) (oX - ovalRadius), (int) (oY - ovalRadius),
                (int) (2 * ovalRadius), (int) (2 * ovalRadius));

        //----------------------------
        brush.setColor(Color.BLACK);
        ovalRadius = (radius - innerRadius) / 15;
        oX = (x - ovalRadius);
        oY = (y - ovalRadius);
        brush.setStroke(new BasicStroke((int) (ovalRadius / 2)));
        brush.drawLine((int) (oX), (int) (oY), (int) (oX - 3 * ovalRadius), (int) (oY));
        brush.rotate(Math.PI / 4, oX, oY);
        brush.drawLine((int) (oX), (int) (oY), (int) (oX - 3 * ovalRadius), (int) (oY));
        brush.rotate(Math.PI / 4, oX, oY);
        brush.drawLine((int) (oX), (int) (oY), (int) (oX - 3 * ovalRadius), (int) (oY));
        brush.rotate(-Math.PI / 2, oX, oY);

        brush.fillOval((int) (oX - ovalRadius), (int) (oY - ovalRadius),
                (int) (2 * ovalRadius), (int) (2 * ovalRadius));

        brush.setColor(Color.CYAN);
        brush.fillRect((int) (oX - ovalRadius / 2), (int) (oY - ovalRadius / 6),
                (int) (2 * ovalRadius / 2), (int) (2 * ovalRadius / 6));

        brush.fillRect((int) (oX - ovalRadius / 6), (int) (oY - ovalRadius / 2),
                (int) (2 * ovalRadius / 6), (int) (2 * ovalRadius / 2));

    }
    */
    /*
    public static void Icon_CogwheelBase(double x, double y, Graphics2D brush, double birthScale, double animScale) {

        double radius =
                (OUTER_REFERENCE_RADIUS * animScale) / birthScale;
        double innerRadius =
                (INNER_REFERENCE_RADIUS * animScale) / birthScale;

        double ovalRadius =
                ((radius - innerRadius) / 4);

        //brush.setColor(Color.BLACK);
        double oX = x;
        double oY = y;

        //brush.fillRect((int)(x-ovalRadius), (int)(y-ovalRadius), (int)(2*ovalRadius), (int)(2*ovalRadius));

        Area clipArea = new Area(new Rectangle.Double((x - radius), (y - radius), (2 * radius), (2 * radius)));
        clipArea.subtract(new Area(new Ellipse2D.Double((oX - ovalRadius * 0.7), (oY - ovalRadius * 0.7), (2 * ovalRadius * 0.7), (2 * ovalRadius * 0.7))));
        brush.setClip(clipArea);

        Polygon p = new Polygon();
        p.addPoint((int) (oX - ovalRadius / 4), (int) (oY + ovalRadius / 10));
        p.addPoint((int) (oX - ovalRadius / 7), (int) (oY - ovalRadius / 3));
        p.addPoint((int) (oX + ovalRadius / 7), (int) (oY - ovalRadius / 3));
        p.addPoint((int) (oX + ovalRadius / 4), (int) (oY + ovalRadius / 10));
        p.translate(0, (int) (-ovalRadius));
        int max = 9;
        for (int i = 0; i < max; i++) {
            brush.rotate((2 * Math.PI / max), oX, oY);
            brush.fillPolygon(p);
        }
        brush.fillOval((int) (oX - ovalRadius), (int) (oY - ovalRadius),
                (int) (2 * ovalRadius), (int) (2 * ovalRadius));

        brush.setClip(null);

    }
    */

    public static void Icon_SigmoidBase(double x, double y, Graphics2D brush, double birthScale, double animScale) {

        double radius = (OUTER_REFERENCE_RADIUS * animScale) / birthScale;
        double innerRadius = (INNER_REFERENCE_RADIUS * animScale) / birthScale;

        Area clipArea;
        double r = (radius - innerRadius) / 2.5;
        x += 0.05*r;
        y += 0.05*r;
        brush.fillRect((int) (x - r * 1.25), (int) (y - 1.25 * r / 9),
                (int) (2 * r * 1.25), (int) (2 * 1.25 * r / 9));
        brush.fillRect((int) (x - 1.25 * r / 9), (int) (y - r * 1.25),
                (int) (2 * 1.25 * r / 9), (int) (2 * r * 1.25));

        double cX = x - r * 0.2;
        double cY = y - r + r * 0.1;
        clipArea = new Area(new Rectangle.Double((x), (y - r), (r), (r)));
        clipArea.subtract(new Area(new Ellipse2D.Double((cX + r * 0.375), (cY + 0.175 * r), (2 * r + 0.4 * r), (2 * r - 0.35 * r))));
        brush.setClip(clipArea);
        //brush.setColor(Color.BLUE);
        brush.fillOval((int) (cX), (int) (cY), (int) (2 * r + r * 0.4), (int) (2 * r));
        //brush.rotate(-Math.PI, x, y);
        cX = x - 2 * r - r * 0.2;
        cY = y - r - r * 0.1;
        clipArea = new Area(new Rectangle.Double((x - r), (y), (r), (r)));
        clipArea.subtract(new Area(new Ellipse2D.Double((cX - r * 0.4), (cY + 0.2 * r), (2 * r + r * 0.4), (2 * r - 0.4 * r))));
        brush.setClip(clipArea);
        brush.fillOval((int) (cX), (int) (cY), (int) (2 * r + r * 0.4), (int) (2 * r));
        brush.setClip(null);
    }

    /*
    //------------------------------------------------x
    public static void Icon_TanhNode(double x, double y, Graphics2D brush, double birthScale, double animScale) {
        double radius = (OUTER_REFERENCE_RADIUS * animScale) / birthScale;
        double innerRadius = (INNER_REFERENCE_RADIUS * animScale) / birthScale;
        double ovalRadius = (radius - innerRadius) / 5;
        brush.setColor(Color.BLACK);
        double oX = x + ovalRadius / 2.55;
        double oY = y + ovalRadius / 2.55;

        Icon_DefaultNode(x, y, brush, birthScale, 1.0);

        double scaleHolder = birthScale;
        birthScale *= (4.5);
        brush.setColor(Color.BLACK);
        Icon_SigmoidBase(oX, oY, brush, birthScale, 1.0);
        birthScale = scaleHolder;

    }
    */


}









