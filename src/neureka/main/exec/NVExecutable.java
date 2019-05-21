package neureka.main.exec;

import java.math.BigInteger;

public interface NVExecutable 
{
	public boolean loadLatest();
	public boolean loadLatest(BigInteger Hi);
	public void    forward();
	
	public boolean loadTrainableState(BigInteger Hi);
	public void    Backward(BigInteger Hi);
	//=====================================
	
}
