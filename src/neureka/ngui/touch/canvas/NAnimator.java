package neureka.ngui.touch.canvas;

import java.util.ArrayList;
import java.util.HashMap;

public class NAnimator {

	//ANIMATED NPANEL ownerS AND THEIR COMPONENTS!
	//A PANEL owner CAN ADD AN ANIMATION COUNTER AN UPDATE IT.
	
	//EVERY LIST MEMBER HAS A CORRESPONDING COUNTER (Integer)
	//-> A PANEL NODE CAN HAVE MULIPLE ANIMATIONS!
	private HashMap<Object, ArrayList<Animation>> AnimationMap;
	//=======================================================================================
	
	NAnimator()
	{AnimationMap = new HashMap<Object, ArrayList<Animation>>();}
	//=======================================================================================
	//---------------------------------------------------------------------------------------
	//=======================================================================================
	public void setCounterFor(Object owner, int startValue)
	{
		setCounterFor(owner, 0, startValue);
	}
	//---------------------------------------------------------------------------------------
	public void setCounterFor(Object owner, int counterID, int startValue)//...
	{
		System.out.println("setting counter: id-"+counterID+", value-"+startValue+"; ");
		if(AnimationMap.containsKey(owner)==false) 
		{
			ArrayList<Animation> animations = new ArrayList<Animation>();
			animations.add(new Animation(counterID, startValue));
			AnimationMap.put(owner, animations);
		}
		else 
		{
			AnimationMap.get(owner).add(new Animation(counterID, startValue));
	 	}
	
	}
	//=======================================================================================
	//---------------------------------------------------------------------------------------
	//=======================================================================================
	public void countDownFor(Object owner) 
	{
		countDownFor(owner, 0);
	}
	//---------------------------------------------------------------------------------------
	public void countDownFor(Object owner, int counterID) 
	{
	    if(AnimationMap.containsKey(owner)) 
	    {
	    	ArrayList<Animation> List = AnimationMap.get(owner);//.setInto(counterID, AnimationMap.getFrom(owner).getFrom(counterID)-1);
	    	for(int Ai=0; Ai<List.size(); Ai++) 
	    	{
	    		if(List.get(Ai).ID==counterID) {List.get(Ai).Counter -=1;}
	    	}
	    }
	}
	public void countDownFor(Object owner, int value, int counterID) 
	{
	     if(AnimationMap.containsKey(owner)) 
	     {
	    	 ArrayList<Animation> List = AnimationMap.get(owner);//.setInto(counterID, AnimationMap.getFrom(owner).getFrom(counterID)-1);
	     	 for(int Ai=0; Ai<List.size(); Ai++) 
	     	 {
	     		 if(List.get(Ai).ID==counterID) {List.get(Ai).Counter -=value;}
	     	 }
	     }
	}
	//=======================================================================================
	//---------------------------------------------------------------------------------------
	//=======================================================================================
	public void countUpFor(Object owner) 
	{
		countUpFor(owner, 0);
	}
	//---------------------------------------------------------------------------------------
	public void countUpFor(Object owner, int counterID) 
	{
	     if(AnimationMap.containsKey(owner)) 
	     {
	    	 ArrayList<Animation> List = AnimationMap.get(owner);//.setInto(counterID, AnimationMap.getFrom(owner).getFrom(counterID)-1);
	      	 for(int Ai=0; Ai<List.size(); Ai++) 
	      	 {
	      		 if(List.get(Ai).ID==counterID) {List.get(Ai).Counter +=1;}
	      	 }
	     }
	}
	public void countUpFor(Object owner, int value, int counterID) 
	{
	     if(AnimationMap.containsKey(owner)) 
	     {
	    	 ArrayList<Animation> List = AnimationMap.get(owner);//.setInto(counterID, AnimationMap.getFrom(owner).getFrom(counterID)-1);
	      	 for(int Ai=0; Ai<List.size(); Ai++) 
	      	 {
	      		 if(List.get(Ai).ID==counterID) {List.get(Ai).Counter +=value;}
	      	 }
	     }
	}
	//=======================================================================================
	//---------------------------------------------------------------------------------------
	//=======================================================================================
	public int getCounterOf(Object owner) 
	{
		return getCounterOf(owner, 0);
	}
	//---------------------------------------------------------------------------------------
	public int getCounterOf(Object owner, int counterID) 
	{
		 if(AnimationMap.containsKey(owner)) 
		 {
			 ArrayList<Animation> List = AnimationMap.get(owner);//.setInto(counterID, AnimationMap.getFrom(owner).getFrom(counterID)-1);
		     for(int Ai=0; Ai<List.size(); Ai++) 
		     {
		    	 if(List.get(Ai).ID==counterID) {return List.get(Ai).Counter;}
		     }
		 }
		 return -1;
	}
	//=======================================================================================
	//---------------------------------------------------------------------------------------
	//=======================================================================================
	public void removeCounterOf(Object owner) 
	{
	    removeCounterOf(owner, 0);
	}
	//---------------------------------------------------------------------------------------
	public void removeCounterOf(Object owner, int counterID) 
	{

		if(AnimationMap.containsKey(owner)) 
		{
			ArrayList<Animation> List = AnimationMap.get(owner);//.setInto(counterID, AnimationMap.getFrom(owner).getFrom(counterID)-1);
		    for(int Ai=0; Ai<List.size(); Ai++) 
		    {
		    	if(List.get(Ai).ID==counterID) {List.remove(Ai);}
		    }
		}
	}
	//=======================================================================================
	//---------------------------------------------------------------------------------------
	public boolean hasCounter(Object owner, int counterID) 
	{
		if(AnimationMap.containsKey(owner)) 
		{
			ArrayList<Animation> List = AnimationMap.get(owner);//.setInto(counterID, AnimationMap.getFrom(owner).getFrom(counterID)-1);
		    for(int Ai=0; Ai<List.size(); Ai++) 
		    {
		    	if(List.get(Ai).ID==counterID) {return true;}
		    }
		}
		return false;
	}


	private class Animation
	{	
		public int ID;
		public int Counter;
		Animation(int id, int counter)
		{
			ID = id;
			Counter = counter;
		}
		
	}
}
