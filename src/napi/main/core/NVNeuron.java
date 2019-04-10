package napi.main.core;

import napi.main.core.comp.NVData;
import napi.main.core.comp.NVElement;

public interface NVNeuron 
{

	public boolean memorize(NVData Data);
	
	public NVData activationOf(NVData Data, int Vi);
	
	public NVData weightedConvectionOf(boolean[] inputSignal, int Vi);

	public boolean[] publicized(boolean[] signal);
	
	public boolean[] saved(boolean[] signal);
	
	public boolean[] inputSignalIf(boolean condition);
	
	public boolean forwardCondition();
	
	//------------------------------------------------------------------------
	
	
}
