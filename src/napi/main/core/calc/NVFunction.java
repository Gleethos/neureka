
package napi.main.core.calc;

public interface NVFunction {
	
	public NVFunction newBuild(String expression); 
	
	public abstract double activate(double[] input, double[] bias);
	public abstract double activate(double[] input, int j);
	public abstract double activate(double[] input);
	
	public abstract double derive(double[] input, double[] bias, int index);
	public abstract double derive(double[] input, int index, int j);
	public abstract double derive(double[] input, int index);
	
	public abstract boolean dependsOn(int index);
	
	public abstract String expression();


}

/*
 * leave: 1 double array, 1 index; 
 * --> maybe no array -> instead -> array is being recieved through the function?
 * 
 * 
 * branch: 1 compartment array, 
 * 
 * 
 * 
 * 
 * */
 