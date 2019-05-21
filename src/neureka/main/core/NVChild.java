package neureka.main.core;

public interface NVChild 
{
	public NVParent getParent();
	public void     setParent(NVParent mother);
	//TODO make parent switch structurally sound
	//->implement parent references!
}
