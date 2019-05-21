package neureka.main.comp;

import java.math.BigInteger;
import java.util.ArrayList;

import neureka.main.core.NVNode;
import neureka.main.core.NVertex;
import neureka.main.core.imp.NVertex_Root;
import neureka.main.exec.NVExecutable;

public class NGroup implements NVExecutable
{
	
	ArrayList<NGroup> Connection = new ArrayList<NGroup>();
	
	ArrayList<NVertex> Core = new ArrayList<NVertex>();
	
	interface GroupAction {public void apply(NVNode unit);}
	
	NGroup(int size, String function) 
	{
		for(int i=0; i<0; i++) 
		{
			NVertex newCore = new NVertex_Root(1,false, false, i, 1);
			Core.add(newCore);
		}
	}
	// ============================================================================================================================================================================================
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	
	
	public void forEach(GroupAction action) {Core.forEach((unit)->action.apply(unit));}
	
	
	public NVNode[] getUnitVector() {return null;}
	
	public synchronized void disconnect(NGroup connectionNode) 
	{
		for (int Ii = 0; Ii < Connection.size(); Ii++) 
		{
			disconnect(connectionNode, Ii);
		}
		Connection.remove(connectionNode);
	}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public synchronized void disconnect(NGroup connectionNode, int targIi) 
	{
		Core.forEach((core)->{core.disconnect(connectionNode.getUnitVector(), targIi);});
	}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	// CONNECTING TO NODE:
	// ============================================================================================================================================================================================
	public synchronized void connect(NGroup connectionNode) 
	{
		for (int Ii = 0; Ii < Connection.size(); Ii++) 
		{
			connect(connectionNode, Ii);
		}
		Connection.add(connectionNode);
	}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	public synchronized void connect(NGroup connectionNode, int targIi) 
	{
		Core.forEach((core)->{core.connect(connectionNode.getUnitVector(), targIi);});
	}
	// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	public void setFirst(boolean isFirst)         {Core.forEach((core)->core.setFirst(isFirst));}
	public void setLast(boolean isLast)           {Core.forEach((core)->core.setLast(isLast));}
	public void setHidden(boolean isHidden)       {Core.forEach((core)->core.setHidden(isHidden));}
	public void setParalyzed(boolean isParalyzed) {Core.forEach((core)->core.setParalyzed(isParalyzed));}

	@Override
	public boolean loadLatest() {
		Core.forEach((core)->core.asExecutable().loadLatest());
		return false;
	}

	@Override
	public boolean loadLatest(BigInteger Hi) {
		return false;
	}

	@Override
	public void forward() {
		Core.forEach((core)->core.asExecutable().forward());
		
	}

	@Override
	public boolean loadTrainableState(BigInteger Hi) {
		Core.forEach((core)->core.asExecutable().loadTrainableState(Hi));
		return false;
	}

	@Override
	public void Backward(BigInteger Hi) {
		Core.forEach((core)->core.asExecutable().Backward(Hi));
	}
	
}
