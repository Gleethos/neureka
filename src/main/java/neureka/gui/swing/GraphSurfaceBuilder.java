package neureka.gui.swing;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.util.ArrayList;

import neureka.Tsr;
import neureka.function.factory.autograd.GraphNode;

public class GraphSurfaceBuilder
{

    GraphSurface Surface;

    private int pressRadius = 100;

    public SurfaceNode MouseAttachedSurfaceNode;

    public SurfaceNode ConnectionSurfaceNode;

    public SurfaceNodeInput ChosenInputNode;

    public GraphSurface getSurface() {
        return Surface;
    }

    public void addSurfaceObject(SurfaceNode PanelNeuron) {
        if (Surface.getMap() == null) {
            Surface.setMap(new GridSpaceMap(PanelNeuron.getX(), PanelNeuron.getY(), 10000));
        }
        Surface.setMap(Surface.getMap().addAndUpdate(PanelNeuron));
    }

    public GraphSurfaceBuilder(GraphNode source)
    {
        Surface = new GraphSurface();
        SurfaceNode nnode = new SurfaceNode(source, -400, 2500, this);
        addSurfaceObject(nnode);

        Surface.setPreferredSize(new Dimension(500, 500));
        Surface.setBackground(Color.black);
        Surface.setPaintAction(
                (surface, brush) ->
                {
                    brush.setColor(Color.GREEN);
                    brush.fillOval((int) surface.realX(surface.lastSenseX()) - 5, (int) surface.realY(surface.lastSenseY()) - 5, 10, 10);

                    if (this.ChosenInputNode != null) {
                        double x = this.ChosenInputNode.getX();
                        double y = this.ChosenInputNode.getY();
                        brush.setColor(Color.CYAN);
                        brush.setStroke(new BasicStroke(22));
                        brush.drawLine((int) x, (int) y, (int) surface.realX(surface.lastSenseX()), (int) surface.realY(surface.lastSenseY()));
                    }
                }
        );
        Surface.setClickAction(
                (surface) ->
                {
                    int[] Click = surface.getClick();
                    if (Click == null) {
                        return;
                    }
                    int x = Click[0];
                    int y = Click[1];
                    //Goal: find something on panel -> top most thing -> attach if is attachable...
                    surface.getListener().setDragStart(x, y);
                    SurfaceObject found = surface.findObject(surface.realX(x), surface.realY(y), true, null);
                    if (found != null) {
                        if (found.clickedAt(surface.realX(x), surface.realY(y), surface)) {
                            if (found instanceof SurfaceNode) {
                                SurfaceNode PU = (SurfaceNode) found;
                                SurfaceNodeInput node = PU.testFor_AndGet_InputNode(surface.realX(x), surface.realY(y));

                                if (this.ChosenInputNode != null && node == null) {
                                    ((SurfaceNode) found).connect(this.ConnectionSurfaceNode);
                                    //Tsr C2 = this.ConnectionSurfaceNode.getCore().asCore();
                                    //C2._connect(C1, this.InputIndex);
                                }
                                this.ChosenInputNode = node;

                                if (this.ChosenInputNode != null && PU != null) {
                                    this.ConnectionSurfaceNode = PU;
                                    //for (int i = 0; i < PU.getInputNode().size(); i++) {
                                    //    if (PU.getInputNode().get(i) == this.ChosenInputNode) {
                                    //        this.InputIndex = i;
                                    //    }
                                    //}
                                }
                            }
                        }
                    }
                    surface.setClick(null);
                    surface.repaint(0, 0, surface.getWidth(), surface.getHeight());
                }
        );
        Surface.setPressAction(
                (surface) ->
                {
                    int[] Press = surface.getPress();
                    if (Press == null) return;
                    int x = Press[0];
                    int y = Press[1];
                    //surface.setPress(null);
                }
        );
        Surface.setLongPressAction(
                (surface) ->
                {
                    int[] LongPress = surface.getLongPress();
                    if (LongPress == null || surface.getPress() != null) {
                        surface.setPress(null);
                        return;
                    }
                    int x = LongPress[0];
                    int y = LongPress[1];
                    surface.setLongPress(null);
                }
        );
        Surface.setDoubleClickAction(
                (surface) ->
                {
                    int[] DoubleClick = surface.getDoubleClick();
                    if (DoubleClick == null) return;
                    int x = DoubleClick[0];
                    int y = DoubleClick[1];

                    SurfaceObject found = surface.findObject(surface.realX(x), surface.realY(y), true, null);
                    if (found != null) {
                        found.doubleClickedAt(surface.realX(x), surface.realY(y), surface);
                    } else {
                        GraphNode ghnode = new GraphNode(new Tsr(), null, null, null);
                        SurfaceNode Unit = new SurfaceNode(
                                ghnode,
                                surface.realX((double) x),
                                surface.realY((double) y),
                                this
                        );
                        if (surface.getMap() == null) {
                            surface.setMap(new GridSpaceMap(Unit.getX(), Unit.getY(), 10000));
                        }
                        surface.setMap(surface.getMap().addAndUpdate(Unit));
                        Unit.setHasMoved(true);
                    }
                    surface.setDoubleClick(null);
                }
        );
        Surface.setScaleAction(
                (surface) ->
                {
                    if (surface.getScaling() == null) {
                        return;
                    }
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
                }
        );
        Surface.setSenseAction(
                (surface) ->
                {
                    int[] Sense = surface.getSense();
                    if (Sense == null) {
                        return;
                    }
                    int x = Sense[0];
                    int y = Sense[1];
                    surface.setMovement(null);

                    double realX = surface.realX(x);
                    double realY = surface.realY(y);

                    if ((this.MouseAttachedSurfaceNode != null) && surface.isInTouchMode() == false) {
                        //Listener.addDraggPoint(x, y);
                    }
                    if (surface.isInTouchMode()) {
                        this.MouseAttachedSurfaceNode = null;
                    }
                    if (surface.getFocusObject() != null) {
                        surface.getFocusObject().movementAt(realX, realY, surface);
                    }
                    surface.setFocusObject(surface.findObject(realX, realY, true, null));
                    if (this.ChosenInputNode != null) {
                        double cx = this.ChosenInputNode.getX();
                        double cy = this.ChosenInputNode.getY();
                        double sx = surface.realLastSenseX();
                        double sy = surface.realLastSenseY();
                        //SurfaceRepaintSpace space = new SurfaceRepaintSpace(
                        //        (cx + sx) / 2,
                        //        (cy + sy) / 2,
                        //        Math.abs(cx - sx) / 2,
                        //        Math.abs(cy - sy) / 2,
                        //        null
                        //);
                        //surface.layers()[10].add(space);
                    }
                    surface.setLastSenseX(x);
                    surface.setLastSenseY(y);
                }
        );
        Surface.setSwipeAction(
                (surface) ->
                {
                    int[] Swipe = surface.getSwipe();
                    if (Swipe == null) return;
                    if (surface.isInTouchMode()) {
                        surface.setFocusObject(surface.findObject(surface.realX(Swipe[0]), surface.realY(Swipe[1]), true, null));

                        if (surface.getFocusObject() != null) {

                            surface.setLastSenseX(Swipe[2]);
                            surface.setLastSenseY(Swipe[3]);

                            double[] data = {surface.realX(Swipe[0]), surface.realY(Swipe[1]), surface.realX(Swipe[2]), surface.realY(Swipe[3])};
                            surface.getFocusObject().moveDirectional(data, surface);

                            Swipe[0] = Swipe[2];
                            Swipe[1] = Swipe[3];
                            return;
                        }
                        if (surface.getFocusObject() == null) {
                            surface.translatePanel(Swipe[2] - Swipe[0], Swipe[3] - Swipe[1]);
                            Swipe[0] = Swipe[2];
                            Swipe[1] = Swipe[3];
                            surface.setSwipe(null);
                        }
                    } else {
                        if (Swipe.length == 4) {
                            surface.translatePanel(Swipe[2] - Swipe[0], Swipe[3] - Swipe[1]);
                        }
                        Swipe[0] = Swipe[2];
                        Swipe[1] = Swipe[3];
                        surface.setSwipe(null);
                    }
                    surface.setSwipe(null);
                }
        );
        Surface.setDragAction(
                (surface) ->
                {
                    int[] Drag = surface.getDrag();
                    if (Drag == null) return;

                    if (surface.isInTouchMode()) {
                        int[] PressPoint = surface.getListener().PressPoint;
                        if (PressPoint != null) {
                            double distance = Math.pow(Math.pow(PressPoint[1] - Drag[0], 2) + Math.pow(PressPoint[2] - Drag[1], 2), 0.5);
                            double mod = 1 * (Math.pow(distance, (distance)));
                            PressPoint[0] -= 1 * (mod);
                            if (PressPoint[0] < 0) {
                                PressPoint[0] = 0;
                            }
                            if (distance > pressRadius) {
                                //if (newMenu != null) {
                                //    surface.getListener().PressPoint = null;
                                //    Surface.setMap(Surface.getMap().removeAndUpdate(newMenu));
                                //    newMenu = null;
                                //}
                            } else {
                                //if (newMenu != null) {
                                //    newMenu.modifyAnimationState((int) -mod);
                                //}
                            }
                        }
                    } else {

                    }
                    surface.setSwipe(null);
                }
        );
    }


}
