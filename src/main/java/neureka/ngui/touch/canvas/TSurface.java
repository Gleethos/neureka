package neureka.ngui.touch.canvas;


import javafx.scene.Group;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.paint.Paint;
import javafx.scene.shape.ArcType;
import javafx.scene.text.Font;
import javafx.scene.transform.Affine;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.Timer;


public class TSurface extends Group implements TSurface_I {

	private static final long serialVersionUID = 1L;

	// I/O:
	private NListener Listener;
	// Navigation:
	private Affine Scaler = new Affine();
	private Affine Translator = new Affine();

	private ArrayList<Layer> Layers = new ArrayList<>();

	NMap_I PanelMap = null;
	// Animations:
	private NAnimator Animator;

	private int scaleAnimationCounter=60;
	private int lastSenseX;
	private int lastSenseY;

	private Timer GUITimer;
	// Rendering:
	//private ArrayList<NPanelRepaintSpace> RepaintQueue;

	// Sense Focus:
	private NPanelObject FocusObject;

    public NPanelObject getFocusObject() {return FocusObject;}
    public void setFocusObject(NPanelObject object) {FocusObject = object;}

	private double CenterX = (double)getWidth()/2;
	private double CenterY = (double)getHeight()/2;

	private int[] LongPress;
	private int[] Swipe;
	private int[] Click;
	private int[] DoubleClick;
	private int[] Sense;
	private double[] Scaling;
	private int[] Drag;
	private int[] Press;

	private double width = 1000;
	private double height = 500;

	public double getWidth(){
		return this.width;
	}

	public double getHeight(){
		return this.height;
	}

	public int[] getLongPress() {return LongPress;}
	public int[] getSwipe() {return Swipe;}
	public int[] getDoubleClick() {return DoubleClick;}
	public int[] getSense() {return Sense;}
	public double[] getScaling() {return Scaling;}
	public int[] getDrag() {return Drag;}
	public void setDrag(int[] newDragging) {Drag = newDragging;}
	public void setPress(int[] newPress) {Press = newPress;}
	public void setLongPress(int[] newPress) {LongPress = newPress;}
	public int[] getPress() {return Press;}
	public void setClick(int[] newClick) {Click = newClick;}

	public void setScaling(double[] newScaling) {Scaling = newScaling;}
	public void setDoubleClick(int[] newClick) {DoubleClick = newClick;}

	public void setMovement(int[] newMove) {Sense = newMove;}
	public void setSwipe(int[] newSwipe) {Swipe = newSwipe;}

	private boolean touchMode = true;

	private boolean drawRepaintSpaces = false;
	private boolean advancedRendering = false;
	private boolean mapRendering = false;

	private int     Shade=0;
	private boolean shadedClipping=false;

	private long   frameStart;
	private long   frameEnd;
	private int    frameDelta;

	private double fps;
	private double smoothFPS;
	private double rest=0;

	public interface SurfaceAction{void actOn(TSurface panel);}

	public double getCenterX() {return CenterX;}
	public double getCenterY() {return CenterY;}

	public void setCenterX(double value) {CenterX = value;}
	public void setCenterY(double value) {CenterY = value;}

	public NListener getListener()
	{
		return Listener;
	}

	@Override
	public ArrayList<Layer> getLayers() {
		return Layers;
	}

	public double getFPS() {return fps;}
	public double getSmoothedFPS() {return smoothFPS;}

	// Settings:
	public boolean isAntialiasing() {return advancedRendering;}
	public boolean isMaprendering() {return mapRendering;}
	public boolean isClipRendering() {return drawRepaintSpaces;}
	public boolean isClipShadeRendering() {return shadedClipping;}
	public boolean isInTouchMode() {return touchMode;}

	public int getShade() {return Shade;}
	public void setShade(int value) {Shade = value;}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void setPressAction(SurfaceAction action)
	{
		applyPress = action;
	}
	SurfaceAction applyPress = Utility.DefaultPressAction;
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void setLongPressAction(SurfaceAction action)
	{
		applyLongPress = action;
	}
	SurfaceAction applyLongPress = Utility.DefaultLongPressAction;
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void setDoubleClickAction(SurfaceAction action)
	{
		applyDoubleClick = action;
	}
	SurfaceAction applyDoubleClick = Utility.DefaultDoubleClickAction;
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void setClickAction(SurfaceAction action)
	{
		applyClick = action;
	}
	SurfaceAction applyClick = Utility.DefaultClickAction;

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void setScaleAction(SurfaceAction action)
	{
		applyScaling = action;
	}
	SurfaceAction applyScaling = Utility.DefaultScalingAction;

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void setSenseAction(SurfaceAction action)
	{
		applySense = action;
	}
	SurfaceAction applySense = Utility.DefaultSenseAction;
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void setSwipeAction(SurfaceAction action)
	{
		applySwipe = action;
	}
	SurfaceAction applySwipe = Utility.DefaultSwipeAction;
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public void setDragAction(SurfaceAction action)
	{
		applyDrag = action;
	}
	SurfaceAction applyDrag = Utility.DefaultDragAction;
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public interface SurfacePainter{public void actOn(TSurface panel, GraphicsContext brush);}
	public void setPaintAction(SurfacePainter action)
	{
		Painter = action;
	}
	SurfacePainter Painter;
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


//=======================================================================================
	public TSurface()
 	{
 		//this.setSize(200, 200);
		Painter = TSurface.Utility.DefaultPainter;

     	Listener = new NListener(this);
	 	//this.addMouseListener(Listener);
	 	//this.addMouseWheelListener(Listener);
	 	//this.addMouseMotionListener(Listener);
	 	Animator = new NAnimator();
	 	//setBackground(Color.black);
	 	//repaint(0, 0, getWidth(), getHeight());
	 	//GUITimer = new Timer(10, this);
		//GUITimer.start();
		Layers.add(new Layer(0));
		this.getChildren().add(Layers.get(0));
		//Layers.inject(new Layer(1));
		//Layers.inject(new Layer(2));
 	}
 	public void updateAndRedraw()
 	{
	 	nullRepaintQueue();
	 	Layers.forEach((l)->{
	 		l.setRepaintQueue(new ArrayList<>());
		});
	 	//if(RepaintQueue==null) {
	 	//	RepaintQueue = new ArrayList<>();
	 	//}
	 	//RepaintQueue.addAll(update());
		Layers.forEach((l)->{
			l.getRepaintQueue().addAll(update(l));
		});
		//repaint(0, 0, 250, 30);//framerate
		//repaint(0, 0, getWidth(), getHeight());
	 	startRepaintQueue();

	 	frameEnd=System.nanoTime();
	 	frameDelta = (int) (Math.abs((frameEnd-frameStart)));
	 	frameStart=System.nanoTime();//double seconds = (frameDelta);
	 	fps = 1e9/(((double)frameDelta));
	 	smoothFPS = (fps+12*smoothFPS)/13;

	 	if(scaleAnimationCounter>0)
	   {
		 	double scale =
		 	1/Math.pow(1+(1/((double)scaleAnimationCounter+15)), 2);
		 	System.out.println(scale);
		 	this.scaleAt((int)getWidth()/2, (int)getHeight()/2, scale);
		 	scaleAnimationCounter-=1;
		 	//repaint(0, 0, getWidth(), getHeight());
	    }

	 	if(rest<0) {rest=0;}
	 	double delay = fps-80-rest;//-rest;//ratio*1000;
	 	rest=(delay+rest)/2;
	 	//System.out.println(delay);
	 	if(delay<50 && delay>0)
	 	{
	 		//try {Thread.sleep((long)(delay));} catch (InterruptedException e) {e.printStackTrace();}
	 	}
 	}

 	public Affine getScaler(){
		return Scaler;
	}

	public Affine getTranslator(){
		return Translator;
	}
 //================================================================================================================================
 	public int lastSenseX() 	   {return lastSenseX;}
	public int lastSenseY() 	   {return lastSenseY;}
	public void setLastSenseX(int value) {lastSenseX = value;}
	public void setLastSenseY(int value) {lastSenseY = value;}
	public double realLastSenseX() {return lastSenseX;}
	public double realLastSenseY() {return realY(lastSenseY);}
 	//--------------------------------------------------------------------------------------------------------------------------------
 	public void switchTouchMode()         {if(touchMode==false)         {touchMode=true;}        else {touchMode=false;}}
 	public void switchMapRendering()      {if(mapRendering==false)      {mapRendering=true;}     else {mapRendering=false;}}
 	public void switchDrawRepaintSpaces() {if(drawRepaintSpaces==false) {drawRepaintSpaces=true;}else {drawRepaintSpaces=false;}}
 	public void switchAdvancedRendering() {if(advancedRendering==false) {advancedRendering=true;}else {advancedRendering=false;}}
 	public void switchShadedClipping()    {if(shadedClipping==false)    {shadedClipping=true;}   else {shadedClipping=false;}}
 	//--------------------------------------------------------------------------------------------------------------------------------
 	public int getCurrentFrameDelta()
 	{
 		return frameDelta;
 	}
 	//--------------------------------------------------------------------------------------------------------------------------------
 	//public  ArrayList<NPanelRepaintSpace> getRepaintQueue()
 	//{
 	//	return this.RepaintQueue;
 	//}
	//public void setRepaintQueue(ArrayList<NPanelRepaintSpace> newQueue)
	//{
	//	RepaintQueue = newQueue;
	//}
	//--------------------------------------------------------------------------------------------------------------------------------
	private ArrayList<NPanelRepaintSpace> update(Layer layer)
	{
	 	ArrayList<NPanelRepaintSpace> queue = new ArrayList<NPanelRepaintSpace>();
	 	Listener.updateOn(this);
	 	applySwipe.actOn(this);
	 	applyClick.actOn(this);
	 	applyLongPress.actOn(this);
	 	applyDoubleClick.actOn(this);
	 	applyScaling.actOn(this);
	 	applySense.actOn(this);
	 	applyDrag.actOn(this);

	 	if(PanelMap!=null)
	 	{
		 	LinkedList<NPanelObject> killList = new LinkedList<NPanelObject>();
		 	LinkedList<NCircleMenu> menuList = new LinkedList<NCircleMenu>();
		 	LinkedList<NPanelObject> updateList = new LinkedList<NPanelObject>();
		 	Object[] encQueue = {queue};
		 	PanelMap.applyToAll
		 	(
				 (NPanelObject thing)->
				 {
					 updateList.add(thing);
					 if((thing).killable()) {killList.add((thing));}
					 if(thing instanceof NCircleMenu){menuList.add((NCircleMenu)thing);}
					 return true;
				 }
		    );
		 	updateList.forEach((NPanelObject thing)-> ((ArrayList<NPanelRepaintSpace>)encQueue[0]).addAll((thing.updateOn(this))));
		 	menuList.forEach((element)-> queue.addAll(element.updateOn(this)));
		 	killList.forEach((element)-> PanelMap = PanelMap.removeAndUpdate(element));
	 	}
	 return queue;
 	}
	//--------------------------------------------------------------------------------------------------------------------------------
	public void startRepaintQueue()
	{
		for (Layer layer : this.Layers) {
			if(layer.getRepaintQueue()!=null)
			{
				layer.getRepaintQueue().forEach(
						(repaintSpace)->
						{
							if(repaintSpace!=null)
							{
								repaintAndScaleOnscreenArea
										(
												layer,
												realToOnPanelX(repaintSpace.getCenterX()),
												realToOnPanelY(repaintSpace.getCenterY()),
												repaintSpace.getDistanceX(),
												repaintSpace.getDistanceY()
										);
							}
						});
			}
		}

	}
	//--------------------------------------------------------------------------------------------------------------------------------
	public void nullRepaintQueue()
	{
		for (Layer layer : this.Layers) {
			layer.setRepaintQueue(null);
		}
	}
	//--------------------------------------------------------------------------------------------------------------------------------
    //--------------------------------------------------------------------------------------------------------------------------------
    private void repaintAndScaleOnscreenArea(
    		Layer layer,
    		double centerX,
			double centerY,
			double distanceX,
			double distanceY)
	{
		int LTCornerX = (int)(centerX-1-(distanceX*Scaler.getMxx()));
		int LTCornerY = (int)(centerY-1-(distanceY*Scaler.getMyy()));
		int width  =    (int)(2+       2*distanceX*Scaler.getMxx());
		int height =    (int)(2+       2*distanceY*Scaler.getMyy());

		if(LTCornerX < 0) {width+=LTCornerX; LTCornerX=0; }
		if(LTCornerY < 0) {height+=LTCornerY; LTCornerY=0;}
		if(width < 0)  {width=0; }
		if(height < 0) {height=0;}
		if(LTCornerX > getWidth())  {LTCornerX=(int)getWidth();}
		if(LTCornerY > getHeight()) {LTCornerY=(int)getHeight();}
		if(width > getWidth())   {width=(int)getWidth();  }
		if(height > getHeight()) {height=(int)getHeight();}

		//repaint( LTCornerX, LTCornerY, width, height);
		//paintComp(){....
		GraphicsContext brush = layer.getGraphicsContext2D();
		//super.paintComponent(g);
		//Graphics2D brush = (Graphics2D) g;
		if(this.isAntialiasing())
		{
			//brush.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		}
		else
		{
			//brush.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
		}


        brush.setFill(Color.GREEN);
        brush.setStroke(Color.BLUE);
        brush.setLineWidth(5);
        brush.strokeLine(40, 10, 10, 40);
        brush.fillOval(10, 60, 30, 30);
        brush.strokeOval(60, 60, 30, 30);
        brush.fillRoundRect(110, 60, 30, 30, 10, 10);
        brush.strokeRoundRect(160, 60, 30, 30, 10, 10);
        brush.fillArc(10, 110, 30, 30, 45, 240, ArcType.OPEN);
        brush.fillArc(60, 110, 30, 30, 45, 240, ArcType.CHORD);
        brush.fillArc(110, 110, 30, 30, 45, 240, ArcType.ROUND);
        brush.strokeArc(10, 160, 30, 30, 45, 240, ArcType.OPEN);
        brush.strokeArc(60, 160, 30, 30, 45, 240, ArcType.CHORD);
        brush.strokeArc(110, 160, 30, 30, 45, 240, ArcType.ROUND);
        brush.fillPolygon(new double[]{10, 40, 10, 40},
                new double[]{210, 210, 240, 240}, 4);
        brush.strokePolygon(new double[]{60, 90, 60, 90},
                new double[]{210, 210, 240, 240}, 4);
        brush.strokePolyline(new double[]{110, 140, 110, 140},
                new double[]{210, 210, 240, 240}, 4);

		if(this.isClipShadeRendering())
		{
			double newS = this.getShade()%(200*Math.PI);
			//Shade%=200*Math.PI; Shade= Math.abs(Shade);
			int color = (int) Math.abs(((Math.pow(Math.sin((Math.PI/200)*newS), 3)))*200);
			color=(int) Math.abs(color);
			if(color < 0) {color*=-1;}
			brush.setFill(Paint.valueOf(""+color));//.setColor(new Color(color, color, color));
			brush.fillRect(0, 0, this.getWidth(), this.getHeight());
			newS++;
			newS%=200;
			this.setShade((int) newS);
		}
		Font old = brush.getFont();
		brush.setFont(new Font("Tahoma", 22));//Font.font("Bold"),
		brush.setFill(Color.GREEN);
		brush.fillText((int)this.getFPS()+"; "+this.getSmoothedFPS(), 0, brush.getFont().getSize());
		brush.setFont(old);

		if((this.getCenterY()!=((double)this.getHeight())/2)||(this.getCenterX()!=((double)this.getWidth())/2))
		{
			double newCenterX = ((double)this.getWidth())/2;
			double newCenterY = ((double)this.getHeight())/2;
			this.translatePanel((newCenterX-this.getCenterX()), (newCenterY-this.getCenterY()));
			this.setCenterX(newCenterX);
			this.setCenterY(newCenterY);
		}
		brush.transform(this.getScaler());
		brush.transform(this.getTranslator());

		if(this.isMaprendering())
		{
			if(getMap() !=null)
			{
				getMap().paintStructure(brush);
			}
		}
		//========================
		Painter.actOn(this, brush);
		//========================
		if(this.getMap()!=null)
		{
			LinkedList<NPanelObject>[] paintQueue = new LinkedList[9];
			LinkedList[][] enclosed = {paintQueue};

			for(int[] Li= {0}; Li[0]<paintQueue.length; Li[0]++)
			{
				paintQueue[Li[0]] = new LinkedList<NPanelObject>();
			}

			double[] frame =
			{
				 this.realX(0), this.realY(0),
				 this.realX(this.getWidth()), this.realY(this.getHeight())
			};

			this.getMap().applyToAllWithin(frame,
			(thing)->
			{
					LinkedList<NPanelObject>[] layeredQueue = (LinkedList<NPanelObject>[])enclosed[0];
					if(thing.getLayer().getOrder()>=layeredQueue.length)
					{
						LinkedList<NPanelObject>[] newQueue = new LinkedList[thing.getLayer().getOrder()+1];
						for(int Li=0; Li<layeredQueue.length; Li++)
						{newQueue[Li] = layeredQueue[Li];}
						for(int Li=layeredQueue.length; Li<newQueue.length; Li++)
						{newQueue[Li] = new LinkedList<NPanelObject>();}

						layeredQueue = newQueue;
					}
					//layeredQueue[0].addInto(thing);// has connections? is within frame?
					for(int Li=0; Li<layeredQueue.length; Li++)
					{
						if((thing).needsRepaintOnLayer(Li)) {layeredQueue[Li].add(thing);}
					}
					enclosed[0] = layeredQueue;
					return true;
			});
			paintQueue = enclosed[0];

			for(int[] Li= {0}; Li[0]<paintQueue.length; Li[0]++)
			{
				for(Layer l : Layers){
					paintQueue[Li[0]].forEach((current)->current.repaintLayer(l, brush, this));
				}
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
		if(this.isClipRendering())
		{//System.out.println("Drawing repaint spaces!!");
			Layers.forEach((l)->{
				if(l.getRepaintQueue()!=null)
				{
					brush.setFill(Color.WHITE);//setColor(Color.WHITE);
					brush.setStroke(Paint.valueOf("5f"));//new BasicStroke(5f));
					l.getRepaintQueue().forEach(
							(space)-> {
								brush.strokeRect((int)(space.getCenterX()-space.getDistanceX()), (int)(space.getCenterY()-space.getDistanceY()), (int)(space.getDistanceX())*2, (int)(space.getDistanceY())*2);
							});
				}
			});

		}
		//===

	}
	//--------------------------------------------------------------------------------------------------------------------------------
	public void movementAt(int x, int y)
	{
		int[] newSense = new int[2];
		newSense[0]=x;
		newSense[1]=y;
		Sense = newSense;
	}
	//--------------------------------------------------------------------------------------------------------------------------------
	private NPanelObject find(double x, double y, boolean topMost, NPanelObject upToException, NMap_I.MapAction action)
	{
		LinkedList<NPanelObject> List = null;
		if(PanelMap!=null) {
			List = PanelMap.findAllAt(x, y,action);
		}
		if(List==null) {
			return null;
		}
		NPanelObject best = null;
		ListIterator<NPanelObject> Iterator = List.listIterator();
		while(Iterator.hasNext())
		{
			NPanelObject current = Iterator.next();
			if(best==null)
			{
				best = current;
				if(upToException!=null) {
					if(topMost) {if(upToException.getLayer().getOrder()<current.getLayer().getOrder()) {best=null;}}
					else        {if(upToException.getLayer().getOrder()>current.getLayer().getOrder()) {best=null;}}
				}
			} else {
				if(topMost)
				{//-----------------------
					if(upToException!=null)
					{
						if(best!=null)
						{
							if( current.getLayer().getOrder() > best.getLayer().getOrder()
							&& upToException.getLayer().getOrder()>current.getLayer().getOrder())
							{if(current!=upToException) {best = current;}}
						}
						else
						{
							if(upToException.getLayer().getOrder()>current.getLayer().getOrder())
							{
								if(current!=upToException) {best = current;}
							}
						}

					} else {
						if((current).getLayer().getOrder()>best.getLayer().getOrder())
						{best = current;}
					}

				}//-----------------------
				else
				{//-----------------------
					if(upToException!=null) {
						if(best!=null) {
							if( current.getLayer().getOrder() < best.getLayer().getOrder()
							&&  upToException.getLayer().getOrder()<current.getLayer().getOrder() )
							{if(current!=upToException) {best = current;}}
						} else {
							if(upToException.getLayer().getOrder()>current.getLayer().getOrder())
							{if(current!=upToException) {best = current;}}
						}
					} else {
						if((current).getLayer().getOrder()>=best.getLayer().getOrder())
						{best = current;}
					}
				}//-----------------------
			}
		}
		if(upToException!=null && best!=null) {if(best==upToException) {return null;}}
		return best;
	}
	//============================================================
    //--------------------------------------------------------------------------------------------------------------------------------
 	public Object findAnything(double x, double y, boolean topMost, NPanelObject upToException)
 	{
 		NPanelObject something = findObject(x,y,topMost,upToException);
 		if(something!=null) {return something;}
 		return this;
 	}
 	//--------------------------------------------------------------------------------------------------------------------------------
 	public NPanelObject findObject(double x, double y, boolean topMost, NPanelObject upToException)
 	{
 		return find(x,y,topMost,upToException,
 		(NPanelObject element)-> {
			if(element instanceof NPanelObject) {
				if((element).hasGripAt(x, y, this)) {
					return true;
				}//System.out.println("yeahh");
			}
			return false;
		});
 	}
 	//--------------------------------------------------------------------------------------------------------------------------------
 	public double realX(double x) {return ((x)/Scaler.getMxx())-Translator.getTx();}//
 	public double realY(double y) {return ((y)/Scaler.getMyy())-Translator.getTy();}//
 	public double realToOnPanelX(double x) {return ((x+Translator.getTx())*Scaler.getMxx());}//
 	public double realToOnPanelY(double y) {return ((y+Translator.getTy())*Scaler.getMyy());}//

 	//--------------------------------------------------------------------------------------------------------------------------------
	public void draggedBy(int[] Vector){Swipe=Vector;}

	public void translatePanel(double translateX, double translateY)
	{
		System.out.println("translating panel");
		Translator.translate(translateX*1/Scaler.getMxx(), translateY*1/Scaler.getMyy());
		this.Layers.forEach((l)->l.setRepaintQueue(null));
		//repaint(0, 0, getWidth(), getHeight());//Repaint spaces?
	}

	//--------------------------------------------------------------------------------------------------------------------------------
	public void scaleAtReal(double x, double y, double scaleFactor)
	{
		scaleAt((int)realToOnPanelX(x), (int)realToOnPanelY(y), scaleFactor);
	}
	//--------------------------------------------------------------------------------------------------------------------------------
	//Surface interaction:
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override
	public void scaleAt(int x, int y, double scaleFactor)
	{
		double[] newScaling = new double[3];
		newScaling[0] = x;
		newScaling[1] = y;
		newScaling[2] = scaleFactor;
		Scaling = newScaling;
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override
	public void clickedAt(int x, int y)
	{
		int[] newClick = new int[2];
		newClick[0]=x;
		newClick[1]=y;
		Click = newClick;
	}

	public int[] getClick() {return Click;}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override
	public void pressedAt(int x, int y)
	{
		System.out.println("pressed");
		int[] newDrag = new int[2];
		newDrag[0]=x;
		newDrag[1]=y;
		Drag = newDrag;
	}
	@Override
	public void longPressedAt(int x, int y)
	{
		System.out.println("long pressed");
		int[] newLongPress = new int[2];
		newLongPress[0] = x;
		newLongPress[1] = y;
		LongPress = newLongPress;
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override
	public void releasedAt(int x, int y)
	{

	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override
	public double getScale()
	{
		return Scaler.getMxx();
	}
	@Override
	public void doubleClickedAt(int x, int y)
	{
		int[] newDoubleClick = new int[2];
		newDoubleClick[0] = x;
		newDoubleClick[1] = y;
		DoubleClick = newDoubleClick;
	}
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	@Override
	public void draggedAt(int x, int y)
	{
		int[] newDrag = new int[2];
		newDrag[0] = x;
		newDrag[1] = y;
		Drag = newDrag;
	}
	//----------------------------------

	@Override
	public NAnimator getAnimator() {return Animator;}

	@Override
	public int getFrameDelta() {return frameDelta;}

	@Override
	public NMap_I getMap() {return PanelMap;}

	@Override
	public void setMap(NMap_I newMap) {PanelMap = newMap;}


	public static class Utility
	{
		static SurfacePainter DefaultPainter =
		(surface, brush)->
		{
			if(surface.isMaprendering())
			{
				if(surface.getMap() !=null)
				{
					surface.getMap().paintStructure(brush);
				}
			}
			brush.setStroke(Paint.valueOf("GREEN"));//setColor(Color.GREEN);
			brush.fillOval((int)surface.realX(surface.lastSenseX())-5, (int)surface.realY(surface.lastSenseY())-5, 10, 10);
		};
		//-------------------------------------
		static SurfaceAction DefaultPressAction =
		(surface)->
		{
			int[] Press = surface.getPress();
			if(Press==null) {
				return;
			}
			int x = Press[0];
			int y = Press[1];
			System.out.println("Press: X:"+surface.realX((double)x)+"; Y:"+surface.realY((double)y)+";");

			surface.setPress(null);
		};
		//-------------------------------------
		static SurfaceAction DefaultLongPressAction =
		(surface)->
		{
			int[] LongPress = surface.getLongPress();
			if(LongPress==null) {
				return;
			}
			int x = LongPress[0];
			int y = LongPress[1];
			System.out.println("LongPress: X:"+surface.realX((double)x)+"; Y:"+surface.realY((double)y)+";");

			surface.setLongPress(null);
		};
		//-------------------------------------
		static SurfaceAction DefaultClickAction =
	    (surface)->
		{
		   int[] Click = surface.getClick();
		   if(Click==null) {
			   return;
		   }
		   int x = Click[0];
		   int y = Click[1];

		   System.out.println("X:"+surface.realX((double)x)+"; Y:"+surface.realY((double)y)+";");
		   surface.getListener().setDragStart(x, y);
		   NPanelObject found = surface.findObject(surface.realX(x), surface.realY(y), true, null);
		   if(found!=null)
		   {
			   if(found.clickedAt(surface.realX(x), surface.realY(y), surface)) {
				   found.clickedAt(x, y, surface);
			   }
		   }
		   surface.setClick(null);
		//surface.repaint(0, 0, surface.getWidth(), surface.getHeight());
		};
		//-------------------------------------
		static SurfaceAction DefaultDoubleClickAction = (surface)->
		{
			int[] DoubleClick = surface.getDoubleClick();
			if(DoubleClick==null) {return;}

			int x = DoubleClick[0];
			int y = DoubleClick[1];
			System.out.println("Double clicked at: "+x+", "+y);

			NPanelObject found = surface.findObject(surface.realX(x), surface.realY(y), true, null);
			if(found!=null)
			{
				found.doubleClickedAt(surface.realX(x), surface.realY(y), surface);
			}
			else
			{
				//Double clicked in empty space!
			}
			surface.setDoubleClick(null);
		};

		static SurfaceAction DefaultScalingAction =
		(surface)->
		{
			if(surface.getScaling()==null)
			{
				return;
			}
			System.out.println("Default Scaling");

			int x = (int)surface.getScaling()[0];
			int y = (int)surface.getScaling()[1];
			double scaleFactor = surface.getScaling()[2];

			surface.setLastSenseX(x);
			surface.setLastSenseY(y);

			surface.translatePanel(-x, -y);
			surface.getScaler().scale((scaleFactor), (scaleFactor));
			surface.translatePanel(x, y);
			surface.setScaling(null);
			//surface.repaint(0, 0, surface.getWidth(), surface.getHeight());
		};

		static SurfaceAction DefaultSenseAction =
		(surface)->
		{
			int[] Sense = surface.getSense();
			if(Sense==null)
			{
				return;
			}
			int x = Sense[0];
			int y = Sense[1];
			surface.setMovement(null);

			double realX = surface.realX(x);
			double realY = surface.realY(y);

		    //This needs improvement! -> find neuron function needs to be optimized!
		    if(surface.getFocusObject()!=null)
		    {
		    	surface.getFocusObject().movementAt(realX, realY, surface);
		    }
		    //Overlapping nodes need to be considered!
		    surface.setFocusObject(surface.findObject(realX, realY, true, null));
		    if(surface.getFocusObject() instanceof NCircleMenu)
		    {

		    }
		    surface.setLastSenseX(x);
		    surface.setLastSenseY(y);
		};

		static SurfaceAction DefaultSwipeAction =
		(surface)->
		{
			int[] Swipe = surface.getSwipe();
			if(Swipe==null)
			{
				return;
			}
			if(surface.isInTouchMode())
			{
				surface.setFocusObject(surface.findObject(surface.realX(Swipe[0]),surface.realY(Swipe[1]), true, null));

				if(surface.getFocusObject()!=null)
				{
					
					ArrayList<NPanelRepaintSpace> RepaintQueue = surface.getFocusObject().getLayer().getRepaintQueue();
					if(RepaintQueue==null) {RepaintQueue = new ArrayList<NPanelRepaintSpace>();}
						
					surface.setLastSenseX(Swipe[2]);
					surface.setLastSenseY(Swipe[3]);

					double[] data = {surface.realX(Swipe[0]),surface.realY(Swipe[1]), surface.realX(Swipe[2]),surface.realY(Swipe[3])};
					RepaintQueue.addAll(surface.getFocusObject().moveDirectional(data, surface));
					//surface.get.setRepaintQueue(RepaintQueue);
					
					Swipe[0]=Swipe[2];
				    Swipe[1]=Swipe[3];
					Swipe=null; 
					return;
				}
				if(surface.getFocusObject()==null)
				{
				    surface.translatePanel(Swipe[2]-Swipe[0],Swipe[3]-Swipe[1]);
				    Swipe[0]=Swipe[2];
				    Swipe[1]=Swipe[3];
				    surface.setSwipe(null);
				}
			}
			else 
			{
				if(Swipe.length==4) 
				{
					surface.translatePanel(Swipe[2]-Swipe[0],Swipe[3]-Swipe[1]);
				}
				Swipe[0]=Swipe[2];
				Swipe[1]=Swipe[3];
				surface.setSwipe(null);
			}
			surface.setSwipe(null);
		};
		static SurfaceAction DefaultDragAction = 
		(surface)->
		{
			int[] Drag = surface.getSwipe();
			if(Drag==null)
			{
				return;
			}	
			if(surface.isInTouchMode()) 
			{
					
			}
			else 
			{
					
			}
			surface.setSwipe(null);
		};
	}


}
