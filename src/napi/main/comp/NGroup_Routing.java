package napi.main.comp;

import java.math.BigInteger;
import java.util.ArrayList;
import napi.main.core.NVertex;

public class NGroup_Routing extends NGroup
{

	ArrayList<NVertex> Router = new ArrayList<NVertex>();
	
	int nextID = -3;
	
	NGroup_Routing(int size, String function) {super(size, function);}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	
	
	@Override
	public void setParalyzed(boolean isParalyzed) 
	{
		Router.forEach((core)->core.setParalyzed(isParalyzed));
	
		if(isParalyzed==true) 
		{Core.forEach((core)->core.setParalyzed(isParalyzed));nextID=-3;}
		else 
		{
			nextID=-2;
		}
	}
	
	@Override
	public boolean cleanup()
	{
		Router.forEach((core)->core.asExecutable().cleanup());
		Core.forEach((core)->core.asExecutable().cleanup());
		return false;
	}

	@Override
	public void forward() 
	{
		Router.forEach((core)->core.asExecutable().forward());
		Core.forEach((core)->core.asExecutable().forward());
		if(nextID>=0) 
		{Connection.get(nextID).setParalyzed(false); nextID=-3;}
		else if(nextID==-2) 
		{Router.forEach((core)->core.setParalyzed(true));Core.forEach((core)->core.setParalyzed(false));nextID=-1;}
		else if(nextID==-1) 
		{Core.forEach((core)->core.setParalyzed(true));nextID = findBestRoute();}
	}

	@Override
	public boolean preBackward(BigInteger Hi) 
	{
		Router.forEach((core)->core.asExecutable().preBackward(Hi));
		Core.forEach((core)->core.asExecutable().preBackward(Hi));
		return false;
	}

	@Override
	public void Backward(BigInteger Hi) 
	{
		Router.forEach((core)->core.asExecutable().Backward(Hi));
		Core.forEach((core)->core.asExecutable().Backward(Hi));
	}
	
	
	private int findBestRoute() 
	{	
		int[] counter = {0};
		int[] best = {0};
		double[] largest = {0};
		Router.forEach((node)->
		{
			//if(node.getActivation()>largest[0]) {largest[0]=node.getActivation(); best[0]=counter[0];}
			counter[0]++; 
		});
		return best[0];
	}
	
	
	
	
	
}
