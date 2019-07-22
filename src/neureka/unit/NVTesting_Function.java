package neureka.unit;

import neureka.core.T;
import neureka.core.modul.calc.fcomp.Function;
import neureka.core.modul.calc.FunctionFactory;
import neureka.utility.NMessageFrame;

public class NVTesting_Function extends NVTesting{

	NVTesting_Function(NMessageFrame console, NMessageFrame resultConsole)
	{
		super(console, resultConsole);
	}
	
	public int testExpression(String expression, String expected, String description) 
	{
		Function function;// = new FunctionFactory();
		function = new FunctionFactory().newBuild(expression);
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
		Function function;// = new FunctionFactory();
		function = new FunctionFactory().newBuild(expression);
		printSessionStart(description);
		Console.println(bar+" Function: "+expression);
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

	public int testActivation(String expression, T[] input, T expected, String description) {
		Function function;// = new FunctionFactory();
		function = new FunctionFactory().newBuild(expression);
		printSessionStart(description);
		Console.println(bar+" Function: "+expression);
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
	
	
	
	
}
