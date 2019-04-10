package napi.unit;

import java.util.ArrayList;

import napi.main.core.calc.NVFHead;
import napi.main.core.calc.NVFunction;
import napi.utility.NMessageFrame;

public class NVTesting_Function {

	NMessageFrame Console;
	NMessageFrame ResultConsole;
	String bar = "[I]";
	
	NVTesting_Function(NMessageFrame console, NMessageFrame resultConsole)
	{
		Console=console;
		ResultConsole=resultConsole;
	}
	
	public int testExpression(String expression, String expected, String description) 
	{
		NVFunction function = new NVFHead();
		function = function.newBuild(expression);
		Console.println("");
		Console.println(bar+"  Function Build Test: "+description);
		Console.println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
		Console.println(bar+" Input expression: "+expression);
		Console.println(bar+" Expected expression: "+expected);
		Console.println(bar+"");
		Console.println(bar+" function = function.newBuilt("+expression+");");
		String actual = function.expression();
		Console.println(bar+" String actual = function.expression();");
		Console.println(bar+" 'actual':"+actual);
		Console.println(bar+"----------------------------------------------------");
		Console.println(bar+" Result:");
		int result = 0;
		if(actual.equals(expected)) 
		{
			Console.println(bar+" "+actual+" == "+expected+" |:=> Test successful!");
			result=1;
		}
		else
		{
			Console.println(bar+" "+actual+" =!= "+expected+" |:=> Test failed!");
		}
		Console.println(bar+"====================================================\n");
		return result;
	}
	public int testActivation(String expression, double[] input, double expected, String description) {
		NVFunction function = new NVFHead();
		function = function.newBuild(expression);
		Console.println("");
		Console.println(bar+"  Function Activation Test: "+description);
		Console.println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
		Console.println(bar+" Function: "+expression);
		String inputStr = "";
		for(int Ii=0; Ii<input.length; Ii++) {
			inputStr+=input[Ii]+", ";
		}
		Console.println(bar+" Input: "+inputStr);
		Console.println(bar+"");
		Console.println(bar+" function = function.newBuilt("+expression+");");
		Console.println(bar+" function:"+function.expression());
		Console.println(bar+"----------------------------------------------------");
		Console.println(bar+" Result:");
		int result = 0;
		double activation = function.activate(input);
		if(activation == expected) 
		{
			Console.println(bar+" "+activation+" == "+expected+" |:=> Test successful!");
			result=1;
		}
		else
		{
			Console.println(bar+" "+activation+" =!= "+expected+" |:=> Test failed!");
		}
		Console.println(bar+"====================================================\n");
		return result;
	}
	
	
	
	
	
	
}
