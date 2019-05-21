package neureka.main.core;

import neureka.main.core.base.NVData;

public interface NVRoot {
	//------------------------------------------------------
	
	public double[] findRootDerivativesOf(NVNode unit, NVData ParentData, int Vi);
	
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
