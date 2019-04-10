package napi.main.exec;

import java.math.BigInteger;

public interface NVExecutable 
{
	public boolean cleanup();
	public void    forward();
	
	public boolean preBackward(BigInteger Hi);
	public void    Backward(BigInteger Hi);
	//=====================================
	
}
