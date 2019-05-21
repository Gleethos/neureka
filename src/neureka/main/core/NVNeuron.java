package neureka.main.core;

import neureka.main.core.base.NVData;

public interface NVNeuron 
{

	public boolean memorize(NVData Data);
	
	public NVData activationOf(NVData Data, int Vi);
	
	public NVData weightedConvectionOf(NVData Data, boolean[] inputSignal, int Vi);

	public boolean[] publicized(boolean[] signal);
	
	public boolean[] saved(boolean[] signal);
	
	public boolean[] inputSignalIf(boolean condition);
	
	public boolean forwardCondition();
	
	//------------------------------------------------------------------------
	
	
}
