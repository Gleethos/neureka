package napi.main.core;

import java.math.BigInteger;
import java.util.LinkedList;

public interface NVRoot {
	//------------------------------------------------------
	
	public double[] findRootDerivativesOf(NVNode unit, boolean inputDerivative, int Vi);
	
	public double findRootActivation(int Vi);

	public boolean distributeRootSignal(long phaseSignal);

	public int rootStructureDepth(int level);
	
	public boolean isRootConnection(NVNode[] Connection);
	
	//---------------------------------------
	public boolean isParent();
	public NVParent asParent();
	
	public boolean isChild();
	public NVChild  asChild();
}
