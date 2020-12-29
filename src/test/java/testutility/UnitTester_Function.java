package testutility;

import neureka.Tsr;
import neureka.calculus.Function;
import neureka.calculus.assembly.FunctionBuilder;

public class UnitTester_Function extends UnitTester {

	public UnitTester_Function(String name)
	{
		super(name);
	}

	public int testScalarActivation(Function f, double[] inputs, double expected, String description){
		double actual = f.call(inputs);
		printSessionStart(description);
		println(BAR +" inputs : "+stringified(inputs));
		println(BAR +" Expected: "+expected);
		println(BAR +" function : "+f.toString()+"");
		println(BAR +" String actual = function.toString();");
		println(BAR +" [actual]:"+actual);
		println(BAR +"----------------------------------------------------");
		println(BAR +" Result:");
		assertIsEqual(actual, expected);
		return (printSessionEnd()>0)?1:0;
	}

	public int testScalarDerivative(Function f, double[] inputs, int index, double expected, String description){
		double actual = f.derive(inputs, index);
		printSessionStart(description);
		println(BAR +" inputs : "+stringified(inputs));
		println(BAR +" Expected: "+expected);
		println(BAR +" function : "+f.toString()+"");
		println(BAR +" String actual = function.toString();");
		println(BAR +" [actual]:"+actual);
		println(BAR +"----------------------------------------------------");
		println(BAR +" Result:");
		assertIsEqual(actual, expected);
		return (printSessionEnd()>0)?1:0;
	}
	
	public int testExpression(String expression, String expected, String description) 
	{
		Function function;
		function = new FunctionBuilder().build(expression, true);
		printSessionStart(description);
		println(BAR +" Value toString: "+expression);
		println(BAR +" Expected toString: "+expected);
		println(BAR +"");
		println(BAR +" function = function.newBuilt("+expression+");");
		String actual = function.toString();
		println(BAR +" String actual = function.toString();");
		println(BAR +" [actual]:"+actual);
		println(BAR +"----------------------------------------------------");
		println(BAR +" Result:");
		assertIsEqual(actual, expected);

		return (printSessionEnd()>0)?1:0;
	}
	public int testActivation(String expression, double[] input, double expected, String description) {
		Function function;// = new FunctionBuilder();
		function = new FunctionBuilder().build(expression, true);
		printSessionStart(description);
		println(BAR +" Function: "+expression);
		String inputStr = "";
		for( int Ii=0; Ii<input.length; Ii++ ) {
			inputStr+=input[Ii]+", ";
		}
		println(BAR +" Value: "+inputStr);
		println(BAR +"");
		println(BAR +" function = function.newBuilt("+expression+");");
		println(BAR +" function:"+function.toString());
		println(BAR +"----------------------------------------------------");
		println(BAR +" Result:");
		double activation = function.call(input);
		assertIsEqual(activation, expected);
		return (printSessionEnd()>0)?1:0;
	}

	public int testDeriviation(String expression, double[] input, int d,  double expected, String description) {
		Function function;// = new FunctionBuilder();
		function = new FunctionBuilder().build(expression, true);
		printSessionStart(description);
		println(BAR +" Test Deriviation of Function: "+expression);
		String inputStr = "";
		for(int Ii=0; Ii<input.length; Ii++ ) {
			inputStr+=input[Ii]+", ";
		}
		println(BAR +" Value: "+inputStr);
		println(BAR +"");
		println(BAR +" function = function.newBuilt("+expression+");");
		println(BAR +" function:"+function.toString());
		println(BAR +"----------------------------------------------------");
		println(BAR +" Result:");
		double derivative = function.derive(input, d);
		assertIsEqual(derivative, expected);
		return (printSessionEnd()>0)?1:0;
	}

	public int testActivation(String expression, Tsr[] input, Tsr expected, String description) {
		Function function;
		function = new FunctionBuilder().build(expression, true);
		printSessionStart(description);
		println(BAR +" Function: "+expression);
		String inputStr = "";
		for(int Ii=0; Ii<input.length; Ii++ ) {
			inputStr+=input[Ii]+", ";
		}
		println(BAR +" Value: "+inputStr);
		println(BAR +"");
		println(BAR +" function = function.newBuilt("+expression+");");
		println(BAR +" function:"+function.toString());
		println(BAR +"----------------------------------------------------");
		println(BAR +" Result:");
		Tsr activation = function.call(input);
		assertIsEqual(activation.toString(), expected.toString());
		return (printSessionEnd()>0)?1:0;
	}

	public int testDerivative(String expression, Tsr[] input, int d, Tsr expected, String description) {
		Function function;// = new FunctionBuilder();
		function = new FunctionBuilder().build(expression, true);
		printSessionStart(description);
		println(BAR +" Deriviation of Function: "+expression);
		String inputStr = "";
		for( int Ii=0; Ii<input.length; Ii++ ) {
			inputStr+=input[Ii]+", ";
		}
		println(BAR +" Value: "+inputStr);
		println(BAR +"");
		println(BAR +" function = function.newBuilt("+expression+");");
		println(BAR +" function:"+function.toString());
		println(BAR +"----------------------------------------------------");
		println(BAR +" Result:");
		Tsr derivative = function.derive(input, d);
		assertIsEqual(derivative.toString(), expected.toString());
		return (printSessionEnd()>0)?1:0;
	}
	
}
