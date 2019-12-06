package neureka.gui.swing;


import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.AffineTransform;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.ListIterator;

import javax.swing.JPanel;
import javax.swing.Timer;


public class GraphSurface extends JPanel implements Surface, ActionListener {
    // I/O:
    private SurfaceListener Listener;
    // Navigation:
    private AffineTransform Scaler = new AffineTransform();
    private AffineTransform Translator = new AffineTransform();

    //Space search map:
    AbstractSpaceMap PanelMap = null;

    // Animations:
    private neureka.gui.swing.Animator Animator;

    private int scaleAnimationCounter = 60;
    private int lastSenseX;
    private int lastSenseY;

    private Timer GUITimer;

    // Rendering:
    private ArrayList<SurfaceRepaintSpace> RepaintQueue;

    // Sense Focus:
    private SurfaceObject FocusObject;

    public SurfaceObject getFocusObject() {
        return FocusObject;
    }

    public void setFocusObject(SurfaceObject object) {
        FocusObject = object;
    }

    @Override
    protected boolean requestFocusInWindow(boolean temporary) {
        return super.requestFocusInWindow(temporary);
    }

    private double CenterX = getWidth() / 2;
    private double CenterY = getHeight() / 2;

    private int[] LongPress;
    private int[] Swipe;
    private int[] Click;
    private int[] DoubleClick;
    private int[] Sense;
    private double[] Scaling;
    private int[] Drag;
    private int[] Press;

    public int[] getLongPress() {
        return LongPress;
    }

    public int[] getSwipe() {
        return Swipe;
    }

    public int[] getDoubleClick() {
        return DoubleClick;
    }

    public int[] getSense() {
        return Sense;
    }

    public double[] getScaling() {
        return Scaling;
    }

    public int[] getDrag() {
        return Drag;
    }

    public void setDrag(int[] newDragging) {
        Drag = newDragging;
    }

    public void setPress(int[] newPress) {
        Press = newPress;
    }

    public void setLongPress(int[] newPress) {
        LongPress = newPress;
    }

    public int[] getPress() {
        return Press;
    }

    public void setClick(int[] newClick) {
        Click = newClick;
    }

    public void setScaling(double[] newScaling) {
        Scaling = newScaling;
    }

    public void setDoubleClick(int[] newClick) {
        DoubleClick = newClick;
    }

    public void setMovement(int[] newMove) {
        Sense = newMove;
    }

    public void setSwipe(int[] newSwipe) {
        Swipe = newSwipe;
    }

    private boolean touchMode = true;

    private boolean drawRepaintSpaces = true;
    private boolean advancedRendering = false;
    private boolean mapRendering = false;

    private int Shade = 0;
    private boolean shadedClipping = true;

    private long frameStart;
    private int frameDelta;

    private double fps;
    private double smoothFPS;

    public interface SurfaceAction {
        void actOn(GraphSurface panel);
    }

    public AffineTransform getScaler() {
        return Scaler;
    }

    public AffineTransform getTranslator() {
        return Translator;
    }

    public double getCenterX() {
        return CenterX;
    }

    public double getCenterY() {
        return CenterY;
    }

    public void setCenterX(double value) {
        CenterX = value;
    }

    public void setCenterY(double value) {
        CenterY = value;
    }

    public SurfaceListener getListener() {
        return Listener;
    }

    public double getFPS() {
        return fps;
    }

    public double getSmoothedFPS() {
        return smoothFPS;
    }

    // Settings:
    public boolean isAntialiasing() {
        return advancedRendering;
    }

    public boolean isMaprendering() {
        return mapRendering;
    }

    public boolean isClipRendering() {
        return drawRepaintSpaces;
    }

    public boolean isClipShadeRendering() {
        return shadedClipping;
    }

    public boolean isInTouchMode() {
        return touchMode;
    }

    public int getShade() {
        return Shade;
    }

    public void setShade(int value) {
        Shade = value;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public void setPressAction(SurfaceAction action) {
        applyPress = action;
    }

    SurfaceAction applyPress = Utility.DefaultPressAction;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public void setLongPressAction(SurfaceAction action) {
        applyLongPress = action;
    }

    SurfaceAction applyLongPress = Utility.DefaultLongPressAction;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public void setDoubleClickAction(SurfaceAction action) {
        applyDoubleClick = action;
    }

    SurfaceAction applyDoubleClick = Utility.DefaultDoubleClickAction;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public void setClickAction(SurfaceAction action) {
        applyClick = action;
    }

    SurfaceAction applyClick = Utility.DefaultClickAction;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public void setScaleAction(SurfaceAction action) {
        applyScaling = action;
    }

    SurfaceAction applyScaling = Utility.DefaultScalingAction;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public void setSenseAction(SurfaceAction action) {
        applySense = action;
    }

    SurfaceAction applySense = Utility.DefaultSenseAction;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public void setSwipeAction(SurfaceAction action) {
        applySwipe = action;
    }

    SurfaceAction applySwipe = Utility.DefaultSwipeAction;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public void setDragAction(SurfaceAction action) {
        applyDrag = action;
    }

    SurfaceAction applyDrag = Utility.DefaultDragAction;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public interface SurfacePainter {
        public void actOn(GraphSurface panel, Graphics2D brush);
    }

    public void setPaintAction(SurfacePainter action) {
        Painter = action;
    }

    SurfacePainter Painter;
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    //=======================================================================================
    public GraphSurface() {
        //this.setSize(200, 200);
        Painter = GraphSurface.Utility.DefaultPainter;
        Animator = new Animator();
        Listener = new SurfaceListener(this);
        this.addMouseListener(Listener);
        this.addMouseWheelListener(Listener);
        this.addMouseMotionListener(Listener);
        setBackground(Color.black);
        repaint(0, 0, getWidth(), getHeight());
        GUITimer = new Timer(0, this);
        GUITimer.start();
    }

    public void repaintAll() {
        repaint(0, 0, getWidth(), getHeight());
    }

    public void updateAndRedraw()
    {
        nullRepaintQueue();
        if (RepaintQueue == null) RepaintQueue = new ArrayList<>();
        RepaintQueue.addAll(update());
        //repaint(0, 0, 250, 30);//framerate
        startRepaintQueue();

        frameDelta = (int) (Math.abs((System.nanoTime() - frameStart)));
        fps = 1e9 / (((double) frameDelta));
        if(fps>60){
            double time = (double)(fps-60.0)/4;
            try{
                if(time<50){
                    Thread.sleep((long)time);
                }
            } catch (Exception e){

            }
        }
        frameDelta = (int) (Math.abs((System.nanoTime() - frameStart)));
        fps = 1e9 / (((double) frameDelta));
        smoothFPS = (fps + 12 * smoothFPS) / 13;
        frameStart = System.nanoTime();//double seconds = (frameDelta);

        if (scaleAnimationCounter > 0) {
            double scale = 1 / Math.pow(1 + (1 / ((double) scaleAnimationCounter + 15)), 2);
            //System.out.println(scale);
            this.scaleAt(getWidth() / 2, getHeight() / 2, scale);
            scaleAnimationCounter -= 1;
            repaint(0, 0, getWidth(), getHeight());
        }

    }

    //================================================================================================================================
    public int lastSenseX() {
        return lastSenseX;
    }

    public int lastSenseY() {
        return lastSenseY;
    }

    public void setLastSenseX(int value) {
        lastSenseX = value;
    }

    public void setLastSenseY(int value) {
        lastSenseY = value;
    }

    public double realLastSenseX() {
        return realX(lastSenseX);
    }

    public double realLastSenseY() {
        return realY(lastSenseY);
    }

    //--------------------------------------------------------------------------------------------------------------------------------
    public void switchTouchMode() {
        if (touchMode == false) {
            touchMode = true;
        } else {
            touchMode = false;
        }
    }

    public void switchMapRendering() {
        if (mapRendering == false) {
            mapRendering = true;
        } else {
            mapRendering = false;
        }
    }

    public void switchDrawRepaintSpaces() {
        if (drawRepaintSpaces == false) {
            drawRepaintSpaces = true;
        } else {
            drawRepaintSpaces = false;
        }
    }

    public void switchAdvancedRendering() {
        if (advancedRendering == false) {
            advancedRendering = true;
        } else {
            advancedRendering = false;
        }
    }

    public void switchShadedClipping() {
        if (shadedClipping == false) {
            shadedClipping = true;
        } else {
            shadedClipping = false;
        }
    }

    //--------------------------------------------------------------------------------------------------------------------------------
    public int getCurrentFrameDelta() {
        return frameDelta;
    }

    //--------------------------------------------------------------------------------------------------------------------------------
    public ArrayList<SurfaceRepaintSpace> getRepaintQueue() {
        return this.RepaintQueue;
    }

    public void setRepaintQueue(ArrayList<SurfaceRepaintSpace> newQueue) {
        RepaintQueue = newQueue;
    }

    //--------------------------------------------------------------------------------------------------------------------------------
    private ArrayList<SurfaceRepaintSpace> update() {
        ArrayList<SurfaceRepaintSpace> queue = new ArrayList<SurfaceRepaintSpace>();
        Listener.updateOn(this);
        applySwipe.actOn(this);
        applyClick.actOn(this);
        applyLongPress.actOn(this);
        applyDoubleClick.actOn(this);
        applyScaling.actOn(this);
        applySense.actOn(this);
        applyDrag.actOn(this);

        if (PanelMap != null) {
            LinkedList<SurfaceObject> killList = new LinkedList<SurfaceObject>();
            LinkedList<SurfaceObject> updateList = new LinkedList<SurfaceObject>();
            Object[] enclosedQueue = {queue};
            PanelMap.applyToAll
                    (
                            (SurfaceObject thing) ->
                            {
                                updateList.add(thing);
                                if ((thing).killable()) {
                                    killList.add((thing));
                                }
                                return true;
                            }
                    );
            updateList.forEach(
                    (SurfaceObject thing) ->
                    {
                        ((ArrayList<SurfaceRepaintSpace>) enclosedQueue[0]).addAll((thing.updateOn(this)));
                    }
            );
            killList.forEach((element) ->PanelMap = PanelMap.removeAndUpdate(element));
        }
        return queue;
    }

    //--------------------------------------------------------------------------------------------------------------------------------
    public void startRepaintQueue() {
        if (RepaintQueue != null) {
            RepaintQueue.forEach(
                    (repaintSpace) ->
                    {
                        if (repaintSpace != null) {
                            repaintAndScaleOnscreenArea
                                    (
                                            realToOnPanelX(repaintSpace.getCenterX()),
                                            realToOnPanelY(repaintSpace.getCenterY()),
                                            repaintSpace.getDistanceX(),
                                            repaintSpace.getDistanceY()
                                    );
                        }
                    });
        }
    }

    //--------------------------------------------------------------------------------------------------------------------------------
    public void nullRepaintQueue() {
        RepaintQueue = null;
    }

    //--------------------------------------------------------------------------------------------------------------------------------
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D brush = (Graphics2D) g;
        if (this.isAntialiasing()) {
            brush.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        } else {
            brush.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
        }

        if (this.isClipShadeRendering()) {
            double newS = this.getShade() % (200 * Math.PI);

            //Shade%=200*Math.PI; Shade= Math.abs(Shade);
            int color = (int) Math.abs(((Math.pow(Math.sin((Math.PI / 200) * newS), 3))) * 200);
            color = (int) Math.abs(color);
            if (color < 0) {
                color *= -1;
            }
            brush.setColor(new Color(color, color, color));
            brush.fillRect(0, 0, this.getWidth(), this.getHeight());
            newS++;
            newS %= 200;
            this.setShade((int) newS);
        }
        Font old = brush.getFont();
        brush.setFont(new Font("Tahoma", Font.BOLD, 22));
        brush.setColor(Color.GREEN);
        brush.drawString((int) this.getFPS() + "; " + this.getSmoothedFPS(), 0, brush.getFont().getSize());
        brush.setFont(old);

        if ((this.getCenterY() != ((double) this.getHeight()) / 2) || (this.getCenterX() != ((double) this.getWidth()) / 2)) {
            double newCenterX = ((double) this.getWidth()) / 2;
            double newCenterY = ((double) this.getHeight()) / 2;
            this.translatePanel((newCenterX - this.getCenterX()), (newCenterY - this.getCenterY()));
            this.setCenterX(newCenterX);
            this.setCenterY(newCenterY);
        }

        brush.transform(this.getScaler());
        brush.transform(this.getTranslator());

        if (this.isMaprendering()) {
            if (getMap() != null) {
                getMap().paintStructure(brush);
            }
        }
        //========================
        Painter.actOn(this, brush);
        //========================
        if (this.getMap() != null) {
            LinkedList<SurfaceObject>[] paintQueue = new LinkedList[9];
            LinkedList[][] enclosed = {paintQueue};

            for (int[] Li = {0}; Li[0] < paintQueue.length; Li[0]++) {
                paintQueue[Li[0]] = new LinkedList<SurfaceObject>();
            }

            double[] frame =
                    {
                            this.realX(0), this.realY(0),
                            this.realX(this.getWidth()), this.realY(this.getHeight())
                    };

            this.getMap().applyToAllWithin(frame,
                    (thing) ->
                    {
                        LinkedList<SurfaceObject>[] layeredQueue = (LinkedList<SurfaceObject>[]) enclosed[0];
                        if (thing.getLayerID() >= layeredQueue.length) {
                            LinkedList<SurfaceObject>[] newQueue = new LinkedList[thing.getLayerID() + 1];
                            for (int Li = 0; Li < layeredQueue.length; Li++) {
                                newQueue[Li] = layeredQueue[Li];
                            }
                            for (int Li = layeredQueue.length; Li < newQueue.length; Li++) {
                                newQueue[Li] = new LinkedList<SurfaceObject>();
                            }

                            layeredQueue = newQueue;
                        }
                        //layeredQueue[0].addInto(thing);// has connections? is within frame?
                        for (int Li = 0; Li < layeredQueue.length; Li++) {
                            if ((thing).needsRepaintOnLayer(Li)) {
                                layeredQueue[Li].add(thing);
                            }
                        }
                        enclosed[0] = layeredQueue;
                        return true;
                    });
            paintQueue = enclosed[0];

            for (int[] Li = {0}; Li[0] < paintQueue.length; Li[0]++) {
                paintQueue[Li[0]].forEach((current) -> current.repaintLayer(Li[0], brush, this));
                /* Connections - 0
                 * Root Bodies - 1
                 * Root Connections - 2
                 * Basic Root Nodes - 3
                 * Value Root Nodes - 4
                 * Memory Root Nodes - 5
                 * Super Root Nodes - 6
                 * Default Nodes - 7
                 * Menus
                 * */
            }
            //============================================================================================================
        }

        //RENDERING VISUALIZATION:
        if (this.isClipRendering()) {//System.out.println("Drawing repaint spaces!!");
            if (this.getRepaintQueue() != null) {
                brush.setColor(Color.WHITE);
                brush.setStroke(new BasicStroke(5f));
                this.getRepaintQueue().forEach(
                        (space) -> brush.drawRect((int) (space.getCenterX() - space.getDistanceX()), (int) (space.getCenterY() - space.getDistanceY()), (int) (space.getDistanceX()) * 2, (int) (space.getDistanceY()) * 2)
                );
            }
        }
        //===

    }

    //--------------------------------------------------------------------------------------------------------------------------------
    public void movementAt(int x, int y) {
        int[] newSense = new int[2];
        newSense[0] = x;
        newSense[1] = y;
        Sense = newSense;
    }

    //--------------------------------------------------------------------------------------------------------------------------------
    private SurfaceObject find(double x, double y, boolean topMost, SurfaceObject upToException, AbstractSpaceMap.MapAction action) {
        LinkedList<SurfaceObject> List = null;
        if (PanelMap != null) {
            List = PanelMap.findAllAt(x, y, action);
        }
        if (List == null) {
            return null;
        }
        SurfaceObject best = null;
        ListIterator<SurfaceObject> Iterator = List.listIterator();
        while (Iterator.hasNext()) {
            SurfaceObject current = Iterator.next();
            if (best == null) {
                best = current;
                if (upToException != null) {
                    if (topMost) {
                        if (upToException.getLayerID() < current.getLayerID()) {
                            best = null;
                        }
                    } else {
                        if (upToException.getLayerID() > current.getLayerID()) {
                            best = null;
                        }
                    }
                }
            } else {
                if (topMost) {//-----------------------
                    if (upToException != null) {
                        if (best != null) {
                            if (current.getLayerID() > best.getLayerID()
                                    && upToException.getLayerID() > current.getLayerID()) {
                                if (current != upToException) {
                                    best = current;
                                }
                            }
                        } else {
                            if (upToException.getLayerID() > current.getLayerID()) {
                                if (current != upToException) {
                                    best = current;
                                }
                            }
                        }

                    } else {
                        if ((current).getLayerID() > best.getLayerID()) {
                            best = current;
                        }
                    }

                }//-----------------------
                else {//-----------------------
                    if (upToException != null) {
                        if (best != null) {
                            if (current.getLayerID() < best.getLayerID() && upToException.getLayerID() < current.getLayerID()) {
                                if (current != upToException) {
                                    best = current;
                                }
                            }
                        } else {
                            if (upToException.getLayerID() > current.getLayerID()) {
                                if (current != upToException) {
                                    best = current;
                                }
                            }
                        }
                    } else {
                        if ((current).getLayerID() >= best.getLayerID()) {
                            best = current;
                        }
                    }
                }//-----------------------
            }
        }
        if (upToException != null && best != null) {
            if (best == upToException) {
                return null;
            }
        }
        return best;
    }

    //============================================================
    //--------------------------------------------------------------------------------------------------------------------------------
    public Object findAnything(double x, double y, boolean topMost, SurfaceObject upToException) {
        SurfaceObject something = findObject(x, y, topMost, upToException);
        if (something != null) {
            return something;
        }
        return this;
    }

    //--------------------------------------------------------------------------------------------------------------------------------
    public SurfaceObject findObject(double x, double y, boolean topMost, SurfaceObject upToException) {
        return find(x, y, topMost, upToException,
                (SurfaceObject element) -> {
                    if (element instanceof SurfaceObject) {
                        if ((element).hasGripAt(x, y, this)) {
                            return true;
                        }//System.out.println("yeahh");
                    }
                    return false;
                });
    }

    //--------------------------------------------------------------------------------------------------------------------------------
    public double realX(double x) {
        return ((x) / Scaler.getScaleX()) - Translator.getTranslateX();
    }//

    public double realY(double y) {
        return ((y) / Scaler.getScaleY()) - Translator.getTranslateY();
    }//

    public double realToOnPanelX(double x) {
        return ((x + Translator.getTranslateX()) * Scaler.getScaleX());
    }//

    public double realToOnPanelY(double y) {
        return ((y + Translator.getTranslateY()) * Scaler.getScaleY());
    }//

    //--------------------------------------------------------------------------------------------------------------------------------
    public void draggedBy(int[] Vector) {
        Swipe = Vector;
    }

    public void translatePanel(double translateX, double translateY) {
        Translator.translate(translateX * 1 / Scaler.getScaleX(), translateY * 1 / Scaler.getScaleY());
        if (RepaintQueue != null) {
            RepaintQueue = null;
        }
        repaint(0, 0, getWidth(), getHeight());//Repaint spaces?
    }

    //--------------------------------------------------------------------------------------------------------------------------------
    private void repaintAndScaleOnscreenArea(double centerX, double centerY, double distanceX, double distanceY) {
        int LTCornerX = (int) (centerX - 1 - (distanceX * Scaler.getScaleX()));
        int LTCornerY = (int) (centerY - 1 - (distanceY * Scaler.getScaleY()));
        int width = (int) (2 + 2 * distanceX * Scaler.getScaleX());
        int height = (int) (2 + 2 * distanceY * Scaler.getScaleY());

        if (LTCornerX < 0) {
            width += LTCornerX;
            LTCornerX = 0;
        }
        if (LTCornerY < 0) {
            height += LTCornerY;
            LTCornerY = 0;
        }
        if (width < 0) {
            width = 0;
        }
        if (height < 0) {
            height = 0;
        }
        if (LTCornerX > getWidth()) {
            LTCornerX = getWidth();
        }
        if (LTCornerY > getHeight()) {
            LTCornerY = getHeight();
        }
        if (width > getWidth()) {
            width = getWidth();
        }
        if (height > getHeight()) {
            height = getHeight();
        }

        repaint(LTCornerX, LTCornerY, width, height);
    }

    //--------------------------------------------------------------------------------------------------------------------------------
    public void scaleAtReal(double x, double y, double scaleFactor) {
        scaleAt((int) realToOnPanelX(x), (int) realToOnPanelY(y), scaleFactor);
    }

    //--------------------------------------------------------------------------------------------------------------------------------
    //GraphSurface interaction:
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public void scaleAt(int x, int y, double scaleFactor) {
        double[] newScaling = new double[3];
        newScaling[0] = x;
        newScaling[1] = y;
        newScaling[2] = scaleFactor;
        Scaling = newScaling;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public void clickedAt(int x, int y) {
        int[] newClick = new int[2];
        newClick[0] = x;
        newClick[1] = y;
        Click = newClick;
    }

    public int[] getClick() {
        return Click;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @Override
    public void pressedAt(int x, int y) {
        int[] newDrag = new int[2];
        newDrag[0] = x;
        newDrag[1] = y;
        Drag = newDrag;
    }

    @Override
    public void longPressedAt(int x, int y) {
        int[] newLongPress = new int[2];
        newLongPress[0] = x;
        newLongPress[1] = y;
        LongPress = newLongPress;
    }

    @Override
    public void releasedAt(int x, int y) { }

    @Override
    public double getScale() {
        return Scaler.getScaleX();
    }

    @Override
    public void doubleClickedAt(int x, int y) {
        int[] newDoubleClick = new int[2];
        newDoubleClick[0] = x;
        newDoubleClick[1] = y;
        DoubleClick = newDoubleClick;
    }

    @Override
    public void draggedAt(int x, int y) {
        int[] newDrag = new int[2];
        newDrag[0] = x;
        newDrag[1] = y;
        Drag = newDrag;
    }
    @Override
    public Animator getAnimator() {
        return Animator;
    }

    @Override
    public int getFrameDelta() {
        return frameDelta;
    }

    @Override
    public AbstractSpaceMap getMap() {
        return PanelMap;
    }

    @Override
    public void setMap(AbstractSpaceMap newMap) {
        PanelMap = newMap;
    }


    public static class Utility {
        static SurfacePainter DefaultPainter =
                (surface, brush) ->
                {
                    if (surface.isMaprendering()) {
                        if (surface.getMap() != null) {
                            surface.getMap().paintStructure(brush);
                        }
                    }
                    brush.setColor(Color.GREEN);
                    brush.fillOval((int) surface.realX(surface.lastSenseX()) - 5, (int) surface.realY(surface.lastSenseY()) - 5, 10, 10);
                };
        //-------------------------------------
        static SurfaceAction DefaultPressAction =
                (surface) ->
                {
                    int[] Press = surface.getPress();
                    if (Press == null) {
                        return;
                    }
                    int x = Press[0];
                    int y = Press[1];

                    surface.setPress(null);
                };
        //-------------------------------------
        static SurfaceAction DefaultLongPressAction =
                (surface) ->
                {
                    int[] LongPress = surface.getLongPress();
                    if (LongPress == null) {
                        return;
                    }
                    int x = LongPress[0];
                    int y = LongPress[1];

                    surface.setLongPress(null);
                };
        //-------------------------------------
        static SurfaceAction DefaultClickAction =
                (surface) ->
                {
                    int[] Click = surface.getClick();
                    if (Click == null) {
                        return;
                    }
                    int x = Click[0];
                    int y = Click[1];

                    surface.getListener().setDragStart(x, y);
                    SurfaceObject found = surface.findObject(surface.realX(x), surface.realY(y), true, null);
                    if (found != null) {
                        if (found.clickedAt(surface.realX(x), surface.realY(y), surface)) {
                            found.clickedAt(x, y, surface);
                        }
                    }
                    surface.setClick(null);
                    surface.repaint(0, 0, surface.getWidth(), surface.getHeight());
                };
        //-------------------------------------
        static SurfaceAction DefaultDoubleClickAction = (surface) ->
        {
            int[] DoubleClick = surface.getDoubleClick();
            if (DoubleClick == null) {
                return;
            }

            int x = DoubleClick[0];
            int y = DoubleClick[1];

            SurfaceObject found = surface.findObject(surface.realX(x), surface.realY(y), true, null);
            if (found != null) {
                found.doubleClickedAt(surface.realX(x), surface.realY(y), surface);
            } else {
                //Double clicked in empty space!
            }
            surface.setDoubleClick(null);
        };

        static SurfaceAction DefaultScalingAction =
                (surface) ->
                {
                    if (surface.getScaling() == null) return;
                    int x = (int) surface.getScaling()[0];
                    int y = (int) surface.getScaling()[1];
                    double scaleFactor = surface.getScaling()[2];
                    surface.setLastSenseX(x);
                    surface.setLastSenseY(y);
                    surface.translatePanel(-x, -y);
                    surface.getScaler().scale((scaleFactor), (scaleFactor));
                    surface.translatePanel(x, y);
                    surface.setScaling(null);
                    surface.repaint(0, 0, surface.getWidth(), surface.getHeight());
                };

        static SurfaceAction DefaultSenseAction =
                (surface) ->
                {
                    int[] Sense = surface.getSense();
                    if (Sense == null) return;
                    int x = Sense[0];
                    int y = Sense[1];
                    surface.setMovement(null);
                    double realX = surface.realX(x);
                    double realY = surface.realY(y);
                    //This needs improvement! -> find neuron function needs to be optimized!
                    if (surface.getFocusObject() != null) surface.getFocusObject().movementAt(realX, realY, surface);
                    //Overlapping nodes need to be considered!
                    surface.setFocusObject(surface.findObject(realX, realY, true, null));
                    surface.setLastSenseX(x);
                    surface.setLastSenseY(y);
                };

        static SurfaceAction DefaultSwipeAction =
                (surface) ->
                {
                    int[] Swipe = surface.getSwipe();
                    if (Swipe == null) return;
                    if (surface.isInTouchMode()) {
                        surface.setFocusObject(surface.findObject(surface.realX(Swipe[0]), surface.realY(Swipe[1]), true, null));
                        if (surface.getFocusObject() != null) {
                            ArrayList<SurfaceRepaintSpace> RepaintQueue = surface.getRepaintQueue();
                            if (RepaintQueue == null) RepaintQueue = new ArrayList<SurfaceRepaintSpace>();
                            surface.setLastSenseX(Swipe[2]);
                            surface.setLastSenseY(Swipe[3]);
                            double[] data = {surface.realX(Swipe[0]), surface.realY(Swipe[1]), surface.realX(Swipe[2]), surface.realY(Swipe[3])};
                            RepaintQueue.addAll(surface.getFocusObject().moveDirectional(data, surface));
                            surface.setRepaintQueue(RepaintQueue);
                            Swipe[0] = Swipe[2];
                            Swipe[1] = Swipe[3];
                            Swipe = null;
                            return;
                        }
                        if (surface.getFocusObject() == null) {
                            surface.translatePanel(Swipe[2] - Swipe[0], Swipe[3] - Swipe[1]);
                            Swipe[0] = Swipe[2];
                            Swipe[1] = Swipe[3];
                            surface.setSwipe(null);
                        }
                    } else {
                        if (Swipe.length == 4) surface.translatePanel(Swipe[2] - Swipe[0], Swipe[3] - Swipe[1]);
                        Swipe[0] = Swipe[2];
                        Swipe[1] = Swipe[3];
                        surface.setSwipe(null);
                    }
                    surface.setSwipe(null);
                };
        static SurfaceAction DefaultDragAction =
                (surface) ->
                {
                    int[] Drag = surface.getSwipe();
                    if (Drag == null) return;
                    if (surface.isInTouchMode()) {

                    } else {

                    }
                    surface.setSwipe(null);
                };
    }

    @Override
    public void actionPerformed(ActionEvent arg0) {
        if (arg0.getSource() == GUITimer) updateAndRedraw();
    }

}
