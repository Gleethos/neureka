package neureka.main.exec.cpu;

import java.math.BigInteger;
import java.util.LinkedHashMap;


public class NNetworkExecuter {
	
	private LinkedHashMap<NVExecutable, NThreadable> Map = new LinkedHashMap<NVExecutable, NThreadable>(); 

	NThreadPool NetworkThreader = new NThreadPool(Map);
	Thread T = new Thread(NetworkThreader);

	
	
	
	public boolean Execute(BigInteger forward, BigInteger backward) 
	{
		if(T.isAlive()) {return false;}
		NetworkThreader.setForwrad(forward);
		NetworkThreader.setBackward(backward);
		return Execute();
	}
	public boolean Execute() 
	{
		if(T.isAlive()) {return false;}
		T = new Thread(NetworkThreader);
		T.start();
		return false;
	}

	
}
