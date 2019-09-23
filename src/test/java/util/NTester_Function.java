package util;

import neureka.core.T;
import neureka.core.function.IFunction;
import neureka.core.function.factory.assembly.FunctionBuilder;

public class NTester_Function extends NTester {

	public NTester_Function(String name)
	{
		super(name);
	}
	
	public int testExpression(String expression, String expected, String description) 
	{
		IFunction function;// = new FunctionBuilder();
		function = new FunctionBuilder().build(expression, true);
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
		assertIsEqual(actual, expected);

		return (printSessionEnd()>0)?1:0;
	}
	public int testActivation(String expression, double[] input, double expected, String description) {
		IFunction function;// = new FunctionBuilder();
		function = new FunctionBuilder().build(expression, true);
		printSessionStart(description);
		println(bar+" IFunction: "+expression);
		String inputStr = "";
		for(int Ii=0; Ii<input.length; Ii++) {
			inputStr+=input[Ii]+", ";
		}
		println(bar+" Value: "+inputStr);
		println(bar+"");
		println(bar+" function = function.newBuilt("+expression+");");
		println(bar+" function:"+function.toString());
		println(bar+"----------------------------------------------------");
		println(bar+" Result:");
		double activation = function.activate(input);
		assertIsEqual(activation, expected);
		return (printSessionEnd()>0)?1:0;
	}

	public int testDeriviation(String expression, double[] input, int d,  double expected, String description) {
		IFunction function;// = new FunctionBuilder();
		function = new FunctionBuilder().build(expression, true);
		printSessionStart(description);
		println(bar+" Test Deriviation of IFunction: "+expression);
		String inputStr = "";
		for(int Ii=0; Ii<input.length; Ii++) {
			inputStr+=input[Ii]+", ";
		}
		println(bar+" Value: "+inputStr);
		println(bar+"");
		println(bar+" function = function.newBuilt("+expression+");");
		println(bar+" function:"+function.toString());
		println(bar+"----------------------------------------------------");
		println(bar+" Result:");
		double derivative = function.derive(input, d);
		assertIsEqual(derivative, expected);
		return (printSessionEnd()>0)?1:0;
	}

	public int testActivation(String expression, T[] input, T expected, String description) {
		IFunction function;// = new FunctionBuilder();
		function = new FunctionBuilder().build(expression, true);
		printSessionStart(description);
		println(bar+" IFunction: "+expression);
		String inputStr = "";
		for(int Ii=0; Ii<input.length; Ii++) {
			inputStr+=input[Ii]+", ";
		}
		println(bar+" Value: "+inputStr);
		println(bar+"");
		println(bar+" function = function.newBuilt("+expression+");");
		println(bar+" function:"+function.toString());
		println(bar+"----------------------------------------------------");
		println(bar+" Result:");
		T activation = function.activate(input);
		assertIsEqual(activation.toString(), expected.toString());
		return (printSessionEnd()>0)?1:0;
	}


	public int testDerivative(String expression, T[] input, int d, T expected, String description) {
		IFunction function;// = new FunctionBuilder();
		function = new FunctionBuilder().build(expression, true);
		printSessionStart(description);
		println(bar+" Deriviation of IFunction: "+expression);
		String inputStr = "";
		for(int Ii=0; Ii<input.length; Ii++) {
			inputStr+=input[Ii]+", ";
		}
		println(bar+" Value: "+inputStr);
		println(bar+"");
		println(bar+" function = function.newBuilt("+expression+");");
		println(bar+" function:"+function.toString());
		println(bar+"----------------------------------------------------");
		println(bar+" Result:");
		T derivative = function.derive(input, d);
		assertIsEqual(derivative.toString(), expected.toString());
		return (printSessionEnd()>0)?1:0;
	}
	
}
