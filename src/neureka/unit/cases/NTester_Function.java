package neureka.unit.cases;

import neureka.core.T;
import neureka.core.function.TFunction;
import neureka.core.function.factory.worker.FBuilder;

public class NTester_Function extends NTester {

	public NTester_Function(String name)
	{
		super(name);
	}
	
	public int testExpression(String expression, String expected, String description) 
	{
		TFunction function;// = new FBuilder();
		function = new FBuilder().newBuild(expression, true);
		printSessionStart(description);
		println(bar+" Value toString: "+expression);
		println(bar+" Expected toString: "+expected);
		println(bar+"");
		println(bar+" function = function.newBuilt("+expression+");");
		String actual = function.toString();
		println(bar+" String actual = function.toString();");
		println(bar+" [actual]:"+actual);
		println(bar+"----------------------------------------------------");
		println(bar+" Result:");
		assertEqual(actual, expected);

		return (printSessionEnd()>0)?1:0;
	}
	public int testActivation(String expression, double[] input, double expected, String description) {
		TFunction function;// = new FBuilder();
		function = new FBuilder().newBuild(expression, true);
		printSessionStart(description);
		Console.println(bar+" TFunction: "+expression);
		String inputStr = "";
		for(int Ii=0; Ii<input.length; Ii++) {
			inputStr+=input[Ii]+", ";
		}
		Console.println(bar+" Value: "+inputStr);
		Console.println(bar+"");
		Console.println(bar+" function = function.newBuilt("+expression+");");
		Console.println(bar+" function:"+function.toString());
		Console.println(bar+"----------------------------------------------------");
		Console.println(bar+" Result:");
		double activation = function.activate(input);
		assertEqual(activation, expected);
		return (printSessionEnd()>0)?1:0;
	}

	public int testDeriviation(String expression, double[] input, int d,  double expected, String description) {
		TFunction function;// = new FBuilder();
		function = new FBuilder().newBuild(expression, true);
		printSessionStart(description);
		Console.println(bar+" Test Deriviation of TFunction: "+expression);
		String inputStr = "";
		for(int Ii=0; Ii<input.length; Ii++) {
			inputStr+=input[Ii]+", ";
		}
		Console.println(bar+" Value: "+inputStr);
		Console.println(bar+"");
		Console.println(bar+" function = function.newBuilt("+expression+");");
		Console.println(bar+" function:"+function.toString());
		Console.println(bar+"----------------------------------------------------");
		Console.println(bar+" Result:");
		double derivative = function.derive(input, d);
		assertEqual(derivative, expected);
		return (printSessionEnd()>0)?1:0;
	}

	public int testActivation(String expression, T[] input, T expected, String description) {
		TFunction function;// = new FBuilder();
		function = new FBuilder().newBuild(expression, true);
		printSessionStart(description);
		Console.println(bar+" TFunction: "+expression);
		String inputStr = "";
		for(int Ii=0; Ii<input.length; Ii++) {
			inputStr+=input[Ii]+", ";
		}
		Console.println(bar+" Value: "+inputStr);
		Console.println(bar+"");
		Console.println(bar+" function = function.newBuilt("+expression+");");
		Console.println(bar+" function:"+function.toString());
		Console.println(bar+"----------------------------------------------------");
		Console.println(bar+" Result:");
		T activation = function.activate(input);
		assertEqual(activation.toString(), expected.toString());
		return (printSessionEnd()>0)?1:0;
	}


	public int testDerivative(String expression, T[] input, int d, T expected, String description) {
		TFunction function;// = new FBuilder();
		function = new FBuilder().newBuild(expression, true);
		printSessionStart(description);
		Console.println(bar+" Deriviation of TFunction: "+expression);
		String inputStr = "";
		for(int Ii=0; Ii<input.length; Ii++) {
			inputStr+=input[Ii]+", ";
		}
		Console.println(bar+" Value: "+inputStr);
		Console.println(bar+"");
		Console.println(bar+" function = function.newBuilt("+expression+");");
		Console.println(bar+" function:"+function.toString());
		Console.println(bar+"----------------------------------------------------");
		Console.println(bar+" Result:");
		T derivative = function.derive(input, d);
		assertEqual(derivative.toString(), expected.toString());
		return (printSessionEnd()>0)?1:0;
	}
	
}
