package neureka.gui.swing;

import java.util.ArrayList;
import java.util.HashMap;

public class Animator {

	//ANIMATED OBJECTS AND THEIR COMPONENTS!
	//A OWNER CAN ADD AN ANIMATION COUNTER AN UPDATE IT.
	
	//EVERY LIST MEMBER HAS A CORRESPONDING COUNTER (Integer)
	//-> A PANEL NODE CAN HAVE MULTIPLE ANIMATIONS!
	private HashMap<Object, ArrayList<Animation>> _animationMap;
	//=======================================================================================
	
	public Animator()
	{
		_animationMap = new HashMap<>();
	}
	//=======================================================================================
	//---------------------------------------------------------------------------------------
	//=======================================================================================
	public void setCounterFor(Object owner, int counterID, int startValue)//...
	{
		if(_animationMap.containsKey(owner)==false)
		{
			ArrayList<Animation> animations = new ArrayList<Animation>();
			animations.add(new Animation(counterID, startValue));
			_animationMap.put(owner, animations);
		}
		else 
		{
			_animationMap.get(owner).add(new Animation(counterID, startValue));
	 	}
	
	}
	//=======================================================================================
	//---------------------------------------------------------------------------------------
	//=======================================================================================
	public void countDownFor(Object owner, int value, int counterID) 
	{
	     if(_animationMap.containsKey(owner))
	     {
	    	 ArrayList<Animation> List = _animationMap.get(owner);
	     	 for(int Ai=0; Ai<List.size(); Ai++) 
	     	 {
	     		 if(List.get(Ai).ID==counterID) {List.get(Ai).Counter -=value;}
	     	 }
	     }
	}
	//=======================================================================================
	//---------------------------------------------------------------------------------------
	//=======================================================================================
	public void countUpFor(Object owner, int value, int counterID) 
	{
	     if(_animationMap.containsKey(owner))
	     {
	    	 ArrayList<Animation> List = _animationMap.get(owner);
	      	 for(int Ai=0; Ai<List.size(); Ai++) 
	      	 {
	      		 if(List.get(Ai).ID==counterID) {List.get(Ai).Counter +=value;}
	      	 }
	     }
	}
	//=======================================================================================
	//---------------------------------------------------------------------------------------
	//=======================================================================================
	public int getCounterOf(Object owner, int counterID) 
	{
		 if(_animationMap.containsKey(owner))
		 {
			 ArrayList<Animation> List = _animationMap.get(owner);
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
	public void removeCounterOf(Object owner, int counterID) 
	{
		if(_animationMap.containsKey(owner))
		{
			ArrayList<Animation> List = _animationMap.get(owner);
		    for(int Ai=0; Ai<List.size(); Ai++) 
		    {
		    	if(List.get(Ai).ID==counterID) {List.remove(Ai);}
		    }
		}
	}
	//=======================================================================================
	//---------------------------------------------------------------------------------------
	//=======================================================================================
	public boolean hasCounter(Object owner, int counterID) 
	{
		if(_animationMap.containsKey(owner))
		{
			ArrayList<Animation> List = _animationMap.get(owner);
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
