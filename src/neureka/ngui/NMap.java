package neureka.ngui;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.ListIterator;


public class NMap extends NMap_I {

	NMap_I TLNode, TRNode, BLNode, BRNode;
	double X, Y;
	double size;
	LinkedList<NPanelObject> Elements;
	//CREATE : "EDGE MAP"

	NMap(double centerX, double centerY, double size)
	{
		X = centerX;
		Y = centerY;
		this.size = size;
		TLNode=null;
		TRNode=null;
		BLNode=null;
		BRNode=null;
		Elements = null;
	}
	//------------------------------------------------------------------
	
	public void setTLNode(NMap_I node) {
		TLNode = node;
	}
	public void setTRNode(NMap_I node) {
		TRNode = node;
	}
	public void setBLNode(NMap_I node) {
		BLNode = node;
	}
	public void setBRNode(NMap_I node) {
		BRNode = node;
	}
	//------------------------------------------------------------------
	//==============================================================================
	//------------------------------------------------------------------	
	@Override
	public synchronized NMap_I addAndUpdate(NPanelObject object)
	{
		double LP = object.getLeftPeripheral();
		double RP = object.getRightPeripheral();
		double TP = object.getTopPeripheral();
		double BP = object.getBottomPeripheral();
		
		boolean isBranchable=true;
		if     (LP<=X && X<=RP){isBranchable=false;}
		else if(TP<=Y && Y<=BP){isBranchable=false;}

		if(  (Math.max(Math.abs(X-LP),Math.abs(X-RP))>size 
		   || Math.max(Math.abs(Y-TP),Math.abs(Y-BP))>size)) 
		{
			if(RP<(X+size) && BP<(Y+size)) {
				NMap newBranch = new NMap((X-size),(Y-size), 2*size);
				newBranch.setBRNode(this);
				newBranch = (NMap)newBranch.addAndUpdate(object);//Stack fucking overflow?!?
				return newBranch;
			}
			else if(LP>(X-size) && BP<(Y+size)) {
				NMap newBranch = new NMap((X+size),(Y-size), 2*size);
				newBranch.setBLNode(this);
				newBranch = (NMap)newBranch.addAndUpdate(object);
				return newBranch;
			}
			else if(RP<(X+size) && TP>(Y-size)) {
				NMap newBranch = new NMap((X-size),(Y+size), 2*size);
				newBranch.setTRNode(this);
				newBranch = (NMap)newBranch.addAndUpdate(object);
				return newBranch;
			}
			else if(LP>(X-size) && TP>(Y-size)) {
				NMap newBranch = new NMap((X+size),(Y+size), 2*size);
				newBranch.setTLNode(this);
				newBranch = (NMap)newBranch.addAndUpdate(object);
				return newBranch;
			}
		}
		if(isBranchable==false) 
		{
			if(Elements==null) {Elements=new LinkedList<NPanelObject>();}
			Elements.add(object);
			return this;
		}
		if    (RP <= X && BP <= Y) {
			TLNode = add(TLNode, object, false, false);
		}
			else if(LP > X && BP <= Y) {
			TRNode = add(TRNode, object, true, false);
		}
		else if(RP < X && TP >= Y) {
			BLNode = add(BLNode, object, false, true);
		}
		else if(LP >= X && TP > Y) {
			BRNode = add(BRNode, object, true, true);//Stack fucking overflow?!?
		}
		return this;
	}
	private NMap_I add(NMap_I MapNode, NPanelObject newObject, boolean quadrantX, boolean quadrantY)
	{
		if(MapNode!=null) 
		{
			if(MapNode instanceof Leave) 
			{
			   if(MapNode.getCount()>=MAX)
			   {
				   double vecX = size/2; if(quadrantX==false) {vecX*=-1;}
				   double vecY = size/2; if(quadrantY==false) {vecY*=-1;}
				   NMap newBranch = new NMap(X+vecX, Y+vecY, size/2);
				   newBranch = (NMap)newBranch.addAndUpdate(newObject);
				
				   LinkedList<NPanelObject> oldNodes =((Leave)MapNode).getLeaveElements();
				   for(int i=0; i<oldNodes.size(); i++) {newBranch = (NMap)newBranch.addAndUpdate(oldNodes.get(i));}
				   return newBranch;
			   }
			   else 
			   {
				return MapNode.addAndUpdate(newObject);}
			}
			else 
			{return MapNode.addAndUpdate(newObject);}//Stack fucking overflow?!?!
		}
		else 
		{
			MapNode = new Leave(); 
			MapNode = MapNode.addAndUpdate(newObject); 
		}
		return MapNode;
	}
	//==============================================================================
	
	
	//==============================================================================
	//------------------------------------------------------------------
	@Override
	public synchronized NMap_I removeAndUpdate(NPanelObject object)
	{
		//if(object==null) {return this;}
		//Why not check if branchable? -> The size f the object might have changed! -> not on edge anymore
		if(Elements!=null) 
		{
			if(Elements.remove(object))
			{
				if(Elements.size()==0) {Elements=null;}
				return bestSelf();
			}
		}
		double LP = object.getLeftPeripheral();
		double RP = object.getRightPeripheral();
		double TP = object.getTopPeripheral();
		double BP = object.getBottomPeripheral();
		if    (RP <= X && BP <= Y) {
			TLNode = remove(TLNode, object);
		}
		else if(LP > X && BP <= Y) {//
			TRNode = remove(TRNode, object);
		}
		else if(RP < X && TP >= Y) {//	
			BLNode = remove(BLNode, object);
		}
		else if(LP >= X && TP > Y) {//	
			BRNode = remove(BRNode, object);
		}
		return bestSelf();
	}
	
	private synchronized NMap_I bestSelf()
	{
		byte counter=0;
		if(TLNode  ==null) {counter++;}
		if(TRNode  ==null) {counter++;}
		if(BLNode  ==null) {counter++;}
		if(BRNode  ==null) {counter++;}
		if(Elements==null) {counter++;}
		if(counter==5) {return null;}
		//If the node is a branch: return ... else e_get leave elements and put them into an new branch!
		if(counter==4 && Elements==null) 
		{
			if(TLNode!=null) {return TLNode;}
			if(TRNode!=null) {return TRNode;}
			if(BLNode!=null) {return BLNode;}
			if(BRNode!=null) {return BRNode;}
		}
		return this;
	}
	
	private synchronized NMap_I remove(NMap_I MapNode, NPanelObject oldNode)
	{
		if(MapNode!=null) {MapNode = MapNode.removeAndUpdate(oldNode);} 
		return MapNode;
	}
	//==============================================================================
	
	@Override
	public synchronized int getCount() 
	{
		int nodeCount = 0;
		if(Elements!=null) {nodeCount+=Elements.size();}
		return nodeCount;
	}

	@Override
	public synchronized NPanelObject get(int i) 
	{
		int nodeCount = 0;
		int currentCount=0;
		if(Elements!=null) {
			int count = Elements.size();
			if(i<count) 
			   {return Elements.get(i);}
			else
			   {nodeCount+=count;}
		}
		
		if(TLNode!=null) 
		{
			currentCount=TLNode.getCount();
			if(i<(nodeCount+currentCount)) {return TLNode.get(i-nodeCount);}
			nodeCount += currentCount;
		}
		if(TRNode!=null) 
		{
			currentCount=TRNode.getCount();
			if(i<(nodeCount+currentCount)) {return TRNode.get(i-nodeCount);}
			nodeCount += currentCount;
		}
		if(BLNode!=null) 
		{
			currentCount=BLNode.getCount();
			if(i<(nodeCount+currentCount)) {return BLNode.get(i-nodeCount);}
			nodeCount += currentCount;
		}
		if(BRNode!=null) 
		{
			currentCount=BRNode.getCount();
			if(i<(nodeCount+currentCount)) {return BRNode.get(i-nodeCount);}
			nodeCount += currentCount;
		}
		return null;
	}

	@Override
	public synchronized LinkedList<NPanelObject> getAllClosestTo(double x, double y) 
	{	
		LinkedList<NPanelObject> present = Elements;
		if(present==null) {
			if    (x <= X && y <= Y && TLNode!=null) {
				return TLNode.getAllClosestTo(x, y);
			}
			else if(x > X && y <= Y && TRNode!=null) {
				return TRNode.getAllClosestTo(x, y);
			}
			else if(x < X && y >= Y && BLNode!=null) {
				return BLNode.getAllClosestTo(x, y);
			}
			else if(x >= X && y > Y && BRNode!=null) {
				return BRNode.getAllClosestTo(x, y);
			}
		}
		else 
		{
			if    (x <= X && y <= Y && TLNode!=null) {
			    LinkedList<NPanelObject> found = TLNode.getAllClosestTo(x, y);
				if(found!=null) {found.addAll(present);}
				return found;
			}
			else if(x > X && y <= Y && TRNode!=null) {
				LinkedList<NPanelObject> found = TRNode.getAllClosestTo(x, y);
				if(found!=null) {found.addAll(present);}
				return found;
			}
			else if(x < X && y >= Y && BLNode!=null) {
				LinkedList<NPanelObject> found = BLNode.getAllClosestTo(x, y);
				if(found!=null) {found.addAll(present);}
				return found;
			}
			else if(x >= X && y > Y && BRNode!=null) {
				LinkedList<NPanelObject> found = BRNode.getAllClosestTo(x, y);
				if(found!=null) {found.addAll(present);}
				return found;
			}
		}
		return null;
	}

	@Override
	public synchronized LinkedList<NPanelObject> findAllAt(double x, double y, MapAction condition) 
	{
		LinkedList<NPanelObject> someFound = new LinkedList<NPanelObject>();
		if(Elements!=null) 
		{
			someFound.addAll(filter(condition,Elements));
		}
		if     (x <= X && y <= Y) 
			    { 
				    if(TLNode!=null) {someFound.addAll(TLNode.findAllAt(x, y, condition));}
			        return someFound;
			    }
		else if(x > X && y <= Y) 
			    {
			        if(TRNode!=null) {someFound.addAll(TRNode.findAllAt(x, y, condition));}
			        return someFound;
		        }
		else if(x < X && y >= Y) 
				{
			        if(BLNode!=null) {someFound.addAll(BLNode.findAllAt(x, y, condition));}
			        return someFound;
		        }
		else if(x >= X && y > Y) 
		        {
			        if(BRNode!=null) {someFound.addAll(BRNode.findAllAt(x, y, condition));}
				    return someFound;
		        }
		return null;
	}
	private synchronized LinkedList<NPanelObject> filter(MapAction Actor, LinkedList<NPanelObject> List)
	{
		LinkedList<NPanelObject> filteredList = new LinkedList<NPanelObject>();
		List.forEach((current)->{if(Actor.act(current)) {filteredList.add(current);}});
		return filteredList;
	}


	@Override
	public synchronized void paintStructure(Graphics2D brush) {

		Font F = new Font("Tahoma", Font.PLAIN, (int)(size*(0.1)));
		Color C = brush.getColor();
		Color B = Color.ORANGE;
		brush.setFont(F);
		
		if(TLNode!=null) {
		brush.setColor(C);	
		TLNode.paintStructure(brush);brush.fillOval((int)(X-2000), (int)(Y-2000), 2000, 2000);
		brush.setColor(B);
		brush.drawString(TLNode.getCount()+"", (int)(X-1600), (int)(Y-1200));
		}
		if(TRNode!=null) {
		brush.setColor(C);	
		TRNode.paintStructure(brush);brush.fillOval((int)(X     ), (int)(Y-2000), 2000, 2000);
		brush.setColor(B);
		brush.drawString(TRNode.getCount()+"", (int)(X+1000 ), (int)(Y-1200));
		}
		if(BLNode!=null) {
		brush.setColor(C);	
		BLNode.paintStructure(brush);brush.fillOval((int)(X-2000), (int)(Y     ), 2000, 2000);
		brush.setColor(B);
		brush.drawString(BLNode.getCount()+"", (int)(X-1600), (int)(Y+1200 ));
		}
		if(BRNode!=null) {
		brush.setColor(C);	
		BRNode.paintStructure(brush);brush.fillOval((int)(X     ), (int)(Y     ), 2000, 2000);
		brush.setColor(B);
		brush.drawString(BRNode.getCount()+"", (int)(X+1000 ), (int)(Y+1200));
		}
		if(Elements!=null) 
		{
			ListIterator<NPanelObject> ElementIterator = Elements.listIterator();
			while (ElementIterator.hasNext()) 
			{//System.out.println("checking elements...");
				NPanelObject current = ElementIterator.next();
				brush.setColor(Color.GREEN);
				brush.fillOval((int)(current.getX()-1000), (int)(current.getY()-1000), 2000, 2000);
			}
		brush.setColor(Color.RED);
		brush.fillOval((int)(X-1000), (int)(Y-1000), 2000, 2000);	
		brush.setColor(B);
		brush.drawString(this.getCount()+"", (int)(X-300), (int)(Y-50));
		brush.setColor(C);
		}
		brush.setColor(Color.GREEN);
		brush.drawLine((int)(X), (int)(Y-size),(int)(X), (int)(Y+size));
		brush.drawLine((int)(X-size), (int)(Y),(int)(X+size), (int)(Y));
		brush.setColor(Color.BLUE);
		brush.drawRect((int)(X-size), (int)(Y-size), (int)(size*2), (int)(size*2));
	}

	@Override
	public synchronized boolean applyToAll(MapAction Actor) {
		
		boolean[] actorCheck = {false};
		if(Elements!=null) 
		{//System.out.println("Branch node -> apply to all! START");
			Elements.forEach((current)->{
				boolean check = Actor.act(current);
				if(check) {actorCheck[0]=true;}
			});
			//System.out.println("Branch node -> apply to all! END");
		/*
		    ListIterator<NPanelObject> iterator = Elements.listIterator();
		    while(iterator.hasNext()) {
		    	NPanelObject current = iterator.next();
		    	boolean check = Actor.act(current);
				if(check) {actorCheck[0]=true;}
		    }
		   */
		}
		if(TLNode!=null) {if(TLNode.applyToAll(Actor)) {actorCheck[0]=true;}}
		if(TRNode!=null) {if(TRNode.applyToAll(Actor)) {actorCheck[0]=true;}}
		if(BLNode!=null) {if(BLNode.applyToAll(Actor)) {actorCheck[0]=true;}}
		if(BRNode!=null) {if(BRNode.applyToAll(Actor)) {actorCheck[0]=true;}}
		return actorCheck[0];
	}

	@Override
	public synchronized NMap_I addAll(LinkedList<NPanelObject> elements)
	{
		if(elements==null) {return this;}
		NMap_I[] newNode = {this};
		elements.forEach((current)->{newNode[0] = newNode[0].addAndUpdate(current);});
		return newNode[0];
	}

	@Override
	public synchronized LinkedList<NPanelObject> getAll() 
	{
		LinkedList<NPanelObject> List = new LinkedList<NPanelObject>();
		if(Elements!=null) {List.addAll(Elements);}
		if(TLNode!=null) {List.addAll(TLNode.getAll());}
		if(TRNode!=null) {List.addAll(TRNode.getAll());}
		if(BLNode!=null) {List.addAll(BLNode.getAll());}
		if(BRNode!=null) {List.addAll(BRNode.getAll());}
		return List;
	}

	@Override
	public synchronized Object applyAndGet(AdvancedMapAction Actor) 
	{
		Object thing = null;
		if(Elements!=null)
		{
			ListIterator<NPanelObject> ElementIterator = Elements.listIterator();
			while (ElementIterator.hasNext()) 
			{
				NPanelObject current = ElementIterator.next();
				thing=Actor.act(thing, current);
			}
		}
		if(TLNode!=null) {thing = (TLNode.applyAndGet(Actor));}
		if(TRNode!=null) {thing = (TRNode.applyAndGet(Actor));}
		if(BLNode!=null) {thing = (BLNode.applyAndGet(Actor));}
		if(BRNode!=null) {thing = (BRNode.applyAndGet(Actor));}
		return thing;
	}

	@Override
	public synchronized Object applyAndGet(Object Data, AdvancedMapAction Actor) 
	{
		if(Elements!=null) 
		{
			ListIterator<NPanelObject> ElementIterator = Elements.listIterator();
			while (ElementIterator.hasNext()) 
			{
				NPanelObject current = ElementIterator.next();
				Data=Actor.act(Data, current);
			}
		}
		if(TLNode!=null) {Data = (TLNode.applyAndGet(Data,Actor));}
		if(TRNode!=null) {Data = (TRNode.applyAndGet(Data,Actor));}
		if(BLNode!=null) {Data = (BLNode.applyAndGet(Data,Actor));}
		if(BRNode!=null) {Data = (BRNode.applyAndGet(Data,Actor));}
		return Data;
	}

	@Override
	public synchronized void apply(Object Data, AdvancedMapAction Actor) 
	{
		if(Actor.act(this, null)==null) {return;}
		if(Elements!=null)
		{
			Object[] D = {Data};
			Elements.forEach((current)->{D[0]=Actor.act(D[0], current);});
		}
		if(TLNode!=null) {TLNode.apply(Data,Actor);}
		if(TRNode!=null) {TRNode.apply(Data,Actor);}
		if(BLNode!=null) {BLNode.apply(Data,Actor);}
		if(BRNode!=null) {BRNode.apply(Data,Actor);}
	}

	public double getX() {return X;}
	public double getY() {return Y;}
	public double getSize() {return size;}
	

	@Override
	public synchronized LinkedList<NPanelObject> getAllWithin(double[] frame) 
	{
		LinkedList<NPanelObject> List = new LinkedList<NPanelObject>();
		Elements.forEach(
				(o)->
				{
					if(o.getRightPeripheral()>frame[0]&&o.getLeftPeripheral()<frame[2]
					 &&o.getBottomPeripheral()>frame[1]&&o.getTopPeripheral()<frame[3]) 
					{
						List.add(o);
					}
				});
		if(TLNode!=null) {List.addAll(TLNode.getAllWithin(frame));}
		if(TRNode!=null) {List.addAll(TRNode.getAllWithin(frame));}
		if(BLNode!=null) {List.addAll(BLNode.getAllWithin(frame));}
		if(BRNode!=null) {List.addAll(BRNode.getAllWithin(frame));}
		return List;
	}

	@Override
	public synchronized LinkedList<NPanelObject> findAllWithin(double[] frame, MapAction Actor) 
	{
		LinkedList<NPanelObject> List = new LinkedList<NPanelObject>();
		Elements.forEach(
				(o)->
				{
					if(Actor.act(o)) 
					{
						if(o.getRightPeripheral() >frame[0]&&o.getLeftPeripheral()<frame[2]
						 &&o.getBottomPeripheral()>frame[1]&&o.getTopPeripheral() <frame[3]) 
						{
							List.add(o);
						}
					}
				});
		if(TLNode!=null) {List.addAll(TLNode.findAllWithin(frame, Actor));}
		if(TRNode!=null) {List.addAll(TRNode.findAllWithin(frame, Actor));}
		if(BLNode!=null) {List.addAll(BLNode.findAllWithin(frame, Actor));}
		if(BRNode!=null) {List.addAll(BRNode.findAllWithin(frame, Actor));}
		return List;
	}
	
	@Override
	public synchronized boolean applyToAllWithin(double[] frame, MapAction Actor) 
	{
		boolean[] check = {false};
		if(Elements!=null)
		{
				Elements.forEach(
				(o)->
				{
					if(o.getRightPeripheral()> frame[0]&&o.getLeftPeripheral()<frame[2]
					 &&o.getBottomPeripheral()>frame[1]&&o.getTopPeripheral() <frame[3]) 
					{
						if(Actor.act(o)) 
						{check[0]=true;}
					}
					
				});
		}
		if(TLNode!=null) {if(TLNode.applyToAllWithin(frame,Actor)) {check[0]=true;}}
		if(TRNode!=null) {if(TRNode.applyToAllWithin(frame,Actor)) {check[0]=true;}}
		if(BLNode!=null) {if(BLNode.applyToAllWithin(frame,Actor)) {check[0]=true;}}
		if(BRNode!=null) {if(BRNode.applyToAllWithin(frame,Actor)) {check[0]=true;}}
		return check[0];
	}
		

	
	//===============================================================================================
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	//=> EdgeLeave
	/*
	 * e_get(i)
	 * e_add(NMap node)
	 * remove(NMap node)
	 * forEach(action){}
	 * 
	 * */
	//===============================================================================================
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	public class Leave extends NMap_I {
		 
		HashMap<NPanelObject, NPanelObject> Elements = null;
		
		Leave(){}
		//-------------------------------------------------------------------
		public synchronized LinkedList<NPanelObject> getLeaveElements() 
		{  
			LinkedList<NPanelObject> List = new LinkedList<NPanelObject>();
			Elements.forEach((k, v) -> {List.add(v);});
			return List;
		}
		//-------------------------------------------------------------------
		//==============================================================================
		//------------------------------------------------------------------
		@Override
		public synchronized NMap_I addAndUpdate(NPanelObject node)
		{
			if(node==null) {return this;}
			if(Elements==null) 
			{
				Elements = new HashMap<NPanelObject,NPanelObject>();
			}
			else 
			{
				if(Elements.size()>=MAX) 
				{
					NMap_I newNode;
					double[] newCenterX= {node.getX()};
					double[] newCenterY= {node.getY()};

					Elements.forEach((k, v) -> 
					{
						newCenterX[0] += v.getX();
						newCenterY[0] += v.getY();
					});
					newCenterX[0] /= Elements.size()+1;
					newCenterY[0] /= Elements.size()+1;
					double[] newSize =  {Math.pow(Math.pow(newCenterX[0]-node.getX(), 2)+Math.pow(newCenterY[0]-node.getY(), 2), 0.5)};

					LinkedList<NPanelObject> List = new LinkedList<NPanelObject>();
					Elements.forEach((k, v) -> 
					{
						List.add(v);
						double distance = Math.pow(Math.pow(newCenterX[0]-v.getX(), 2)+Math.pow(newCenterY[0]-v.getY(), 2), 0.5);
						if(distance>newSize[0]) {newSize[0]=distance;}
					});
					newNode = new NMap(newCenterX[0], newCenterY[0], newSize[0]);
					newNode.addAll(List);
					newNode = newNode.addAndUpdate(node);
					return newNode;//new grid
				}
			}
			Elements.put(node,node);//Elements.e_add(node);
			return this;
		}
		//==============================================================================
		//------------------------------------------------------------------
		@Override
		public synchronized NMap_I removeAndUpdate(NPanelObject node) {
			
			if(Elements!=null) 
			{
				Elements.remove(node);
			}
			if(Elements.size()==0) 
			{
				return null;
			}
			return this;
		}
		//==============================================================================
		@Override
		public synchronized int getCount() 
		{
			if(Elements!=null) 
			{
				return Elements.size();
			}
			return 0;
		}
		//==============================================================================
		@Override
		public synchronized NPanelObject get(int i) 
		{
			if(Elements!=null) 
			{
				return Elements.get(i);
			}
			return null;
		}
		//==============================================================================
		@Override
		public synchronized LinkedList<NPanelObject> getAllClosestTo(double x, double y) 
		{
			return this.getLeaveElements();
		}

		@Override
		public synchronized LinkedList<NPanelObject> findAllAt(double x, double y, MapAction condition) 
		{
			if(Elements==null) {return null;}
			LinkedList<NPanelObject> List = new LinkedList<NPanelObject>();
			Elements.forEach((k, v) -> {if(condition.act(v)) {List.add(v);}});	
			return List;
		}
		//==============================================================================
		@Override
		public synchronized void paintStructure(Graphics2D brush) 
		{
			if(Elements==null) {return;}
			Elements.forEach((k, v) -> 
			{
				brush.drawOval((int)(v.getX()-500), (int)(v.getY()-500), 1000, 1000);
				brush.drawOval((int)(v.getX()-700), (int)(v.getY()-700), 1400, 1400);
				brush.drawOval((int)(v.getX()-1000), (int)(v.getY()-1000), 2000, 2000);
				brush.drawOval((int)(v.getX()-2000), (int)(v.getY()-2000), 4000, 4000);
			});
		}
		//==============================================================================
		@Override
		public synchronized boolean applyToAll(MapAction Actor) 
		{
			if(Elements==null) {return false;}
			boolean[] actorCheck = {false};
			
			Elements.forEach((k, v) -> 
			{
				if(Actor.act(v)) {actorCheck[0]=true;}
			});
			return actorCheck[0];
		}
		//==============================================================================
		@Override
		public synchronized NMap_I addAll(LinkedList<NPanelObject> elements)
		{
			if(elements==null) {return this;}	
			if(Elements==null) 
			   {
					if(elements.size()<=MAX) 
					{
						Elements=new HashMap<NPanelObject,NPanelObject>();
						elements.forEach((v)->{Elements.put(v, v);});
					}
				return this;
			   }
			
			if((elements.size()+Elements.size())>MAX) {return this;}//return new branch!
			
			elements.forEach((v)->{Elements.put(v, v);});
			return this;
		}
		//==============================================================================
		@Override
		public synchronized LinkedList<NPanelObject> getAll() 
		{
			return this.getLeaveElements();
		}
		//==============================================================================
		@Override
		public synchronized Object applyAndGet(AdvancedMapAction Actor) 
		{
			if(Elements==null) {return null;}
			Object[] thing = {null};
			Elements.forEach((k, v) -> 
			{
				thing[0]=Actor.act(thing[0], v);
			});
			return thing;
		}
		//==============================================================================
		@Override
		public synchronized Object applyAndGet(Object Data, AdvancedMapAction Actor) 
		{
			if(Elements==null) {return Data;}
			Object[] enclosed = {Data};
			Elements.forEach((k, v) -> {enclosed[0]=Actor.act(enclosed[0], v);});
			return Data;
		}
		//==============================================================================
		@Override
		public synchronized void apply(Object Data, AdvancedMapAction Actor) 
		{
			if(Elements==null) {return;}
			Object[] enclosed = {Data};
			Elements.forEach((k, v) -> {enclosed[0]=Actor.act(enclosed[0], v);});
		}
		@Override
		public synchronized LinkedList<NPanelObject> getAllWithin(double[] frame) {
			LinkedList<NPanelObject> List = new LinkedList<NPanelObject>();
			Elements.forEach(
					(k, o)->
					{
						if(o.getRightPeripheral()>frame[0]&&o.getLeftPeripheral()<frame[2]
					    &&o.getBottomPeripheral()>frame[1]&&o.getTopPeripheral()<frame[3]) 
						{
							List.add(o);
						}
					});
			return List;
		}
		@Override
		public synchronized LinkedList<NPanelObject> findAllWithin(double[] frame, MapAction Actor) 
		{
			LinkedList<NPanelObject> List = new LinkedList<NPanelObject>();
			Elements.forEach(
					(k, o)->
					{
						if(Actor.act(o))
						{
							if(o.getRightPeripheral()>frame[0]&&o.getLeftPeripheral()<frame[2]
						    &&o.getBottomPeripheral()>frame[1]&&o.getTopPeripheral()<frame[3]) 
							{
								List.add(o);
							}
							
						}
					});
			return List;
		}
		@Override
		public synchronized boolean applyToAllWithin(double[] frame, MapAction Actor) 
		{
			boolean[] check = {false};
			if(Elements!=null) 
			{
				Elements.forEach(
					(k,o)->
					{
						if(o.getRightPeripheral()>frame[0]&&o.getLeftPeripheral()<frame[2]
						 &&o.getBottomPeripheral()>frame[1]&&o.getTopPeripheral()<frame[3]) 
						{
							if(Actor.act(o)) 
							{check[0]=true;}
						}
						
					});
			}
			return check[0];
		}
	}
	//===============================================================================================
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


}




