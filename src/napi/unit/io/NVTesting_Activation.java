package napi.unit.io;

import java.math.BigInteger;

import napi.main.core.NVertex;
import napi.utility.NMessageFrame;

public class NVTesting_Activation {

	NMessageFrame Console;
	NMessageFrame ResultConsole;
	String bar = "[I]";
	
	public NVTesting_Activation(NMessageFrame console, NMessageFrame resultConsole){
		Console=console;
		ResultConsole=resultConsole;
	}
	
	
	public int testNode_ScalarActivation(NVertex core, double[] input, double expected) 
	{
		boolean success = true;
		core.setInput(0,input);
		core.asExecutable().forward();
		core.asExecutable().loadState();
		Console.println("");
		Console.println(bar+"  Activation Test: "+core.getFunction().expression());
		Console.println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
		Console.print(bar+"  Setting input to: ");
		for(double v : input) {Console.print(v+"; ");}
		Console.println("");
		Console.println(bar+"  ActivationAt(BigInteger.ZERO): -> Expected output: "+expected);
		double a = core.ActivationAt(BigInteger.ZERO)[0];
		if(a==expected) 
		{
			Console.println(bar+"  Result "+a+" == expected "+expected+" -> test successful.");
		}
		else 
		{
			Console.println(bar+"  Result is: "+a+" -> test failed!");
			success = false;
		}
		Console.println(bar+"----------------------------------------------------");
		Console.println(bar+"  Activation(): -> Expected output: "+expected);
		a = core.getActivation(0);
		if(a==expected) 
		{
			Console.println(bar+"  Result "+a+" == expected "+expected+" -> test successful.");
		}
		else 
		{
			Console.println(bar+"  Result is: "+a+" -> test failed!");
			success = false;
		}
		Console.println(bar+"====================================================\n");
		if(success==false) {return 0;}
		return 1;
	}
	
	public int testGraph_ScalarActivation(NVertex[] Structure, NVertex head, double expected, String description) 
	{
		for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().loadState();}
		for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().forward();  }
		for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().loadState();}
		boolean success = true;
		Console.println("  "+description);
		Console.println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
		Console.println(bar+"  Root Structure Activation Test: (only Vi:0)");
		for(int i=0; i<Structure.length; i++) 
		{
			String expression = "null"; 
			if(Structure[i].getFunction()!=null) 
			{
				expression = Structure[i].getFunction().expression();
			}
			Console.print(bar+"  Root ["+i+"]: ");
			if(Structure[i]==head) {Console.println(bar+"  "+expression+"  => (head)");}
			else {Console.println(expression);}
		}
		
		Console.println("[O]=======================================================");
		Console.println(bar+"  ActivationAt(BigInteger.ZERO): -> Expected output: "+expected);
		double a = head.ActivationAt(BigInteger.ZERO)[0];
		if(a==expected) {Console.println(bar+"  Result "+a+" == expected "+expected+" -> test successful.");}
		else 
		{
			Console.println(bar+"  Result is: "+a+" -> test failed!");
			success = false;
		}
		Console.println("[O]----------------------------------------------------");
		Console.println(bar+"  Activation(): -> Expected output: "+expected);
		Console.println("[O]----------------------------------------------------");
		a = head.getActivation(0);
		if(a==expected) 
		{
			Console.println(bar+"  Result "+a+" == expected "+expected+" -> test successful.");
		}
		else 
		{
			Console.println(bar+"   Result is: "+a+" -> test failed!");
			success = false;
		}
		Console.println("[O]=======================================================");
		if(success==false) {return 0;}
		return 1;
	}
	
	
	public int testNode_VectorActivation(NVertex core, double[][] input, double[] expected, String description) 
	{
		boolean success = true;
		core.setInput(input);
		core.asExecutable().forward();
		core.asExecutable().loadState();
		Console.println("");
		Console.println(" "+core.getFunction().expression());
		Console.println(" "+description+"; Activation Test on Head: "+core.getFunction().expression());
		Console.println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
		Console.print(bar+"  expected: ");
		String wanted = "";
		
		for(double wi : expected) 
		{
			Console.print(wi+"; ");
			wanted += wi+", ";
		}
		Console.println(bar+"");
		
		Console.println(bar+"");
		Console.println(bar+"  ActivationAt(BigInteger.ZERO): -> Expected output: "+expected);
		double a[] = core.ActivationAt(BigInteger.ZERO);
		String result = "";
		for(double ai : a) 
		{
			result += ai+", ";
		}
		if(result.equals(wanted)) 
		{
			Console.println(bar+"  Result "+result+" == expected "+wanted+" -> test successful.");
		}
		else
		{
			Console.println(bar+"  Result is: "+result+" != "+wanted+" -> test failed!");
			success = false;
		}
		Console.println(bar+"----------------------------------------------------");
		Console.println(bar+"  Activation(): -> Expected output: "+wanted);
		result = "";
		
		a = core.getActivation();
		for(double ai : a) 
		{
			result += ai+", ";
		}
		if(result.equals(wanted)) 
		{
			Console.println(bar+"  Result "+result+" == expected "+wanted+" -> test successful.");
		}
		else 
		{
			Console.println(bar+"  Result is: "+result+" !=  "+wanted+"  -> test failed!");
			success = false;
		}
		
		Console.println(bar+"====================================================\n");
		if(success==false) {return 0;}
		return 1;
	}
	public int testGraph_VectorActivation(NVertex[] Structure, NVertex head, double[] expected, String description) 
	{
		for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().loadState();}
		for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().forward();  }
		for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().loadState();  }
		boolean success = true;
		Console.println("  "+description);
		Console.println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
		Console.println(bar+"  Root Structure Activation Test: (only Vi:0)");
		for(int i=0; i<Structure.length; i++) 
		{
			String expression = "null"; 
			if(Structure[i].getFunction()!=null) 
			{
				expression = Structure[i].getFunction().expression();
			}
			if(Structure[i]==head) {Console.println(bar+"  Root ["+i+"]: "+expression+"  => (head)");}
			else {Console.println(bar+"  Root ["+i+"]: "+expression+"");}
		}
		Console.println("[O]=======================================================");
		Console.print(bar+"  expected: ");
		String wanted = "";
		for(double wi : expected) 
		{
			Console.print(wi+", ");
			wanted += wi+", ";
		}
		Console.println("");
		Console.println(bar+"  ActivationAt(BigInteger.ZERO): ");
		double[] a = head.ActivationAt(BigInteger.ZERO);
		String result = "";
		for(double ai : a) 
		{
			result += ai+", ";
		}
		if(result.equals(wanted)) 
		{
			Console.println(bar+"  'result':("+result+") == 'expected': ("+wanted+") -> test successful.");
		}
		else 
		{
			Console.println(bar+"  'result':("+result+") != 'expected':("+wanted+") -> test failed!");
			success = false;
		}
		Console.println("[O]----------------------------------------------------");
		Console.println(bar+"  Activation(): -> Expected output: "+expected);
		Console.println("[O]----------------------------------------------------");
		a = head.getActivation();
		result = "";
		for(double ai : a) 
		{
			result += ai+", ";
		}
		if(result.equals(wanted)) 
		{
			Console.println(bar+"  Result "+result+" -> test successful.");
		}
		else 
		{
			Console.println(bar+"   Result is: "+result+" -> test failed!");
			success = false;
		}
		Console.println("[O]=======================================================");
		if(success==false) {return 0;}
		return 1;
	}
	
	public int testGraph_InputDeriviation(NVertex[] Structure, NVertex head, double[] expected) 
	{
		for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().loadState();}
		for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().forward();}
		for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().loadState();  }
		for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().preBackward(BigInteger.ZERO);}
		for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().Backward(BigInteger.ZERO);}
		
		boolean success = true;
		Console.println("");
		Console.println("Root Structure Deriviation Test: ");
		Console.println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
		for(int i=0; i<Structure.length; i++) 
		{
			String expression = "null"; 
			if(Structure[i].getFunction()!=null) 
			{
				expression = Structure[i].getFunction().expression();
			}
			Console.print(bar+"  Root ["+i+"]: ");
			if(Structure[i]==head) {Console.println(expression+"  => (head)");}
			else 
			{
				Console.println(expression);
			}
		}
		Console.println(bar+"====================================================");
		Console.println(bar+"");
		Console.println(bar+"  Input Derivatives: ");
		Console.print(bar+" ");
		double[] a = head.getInputDerivative()[0];
		for(int i=0; i<a.length; i++) 
		{
			Console.print(a[i]+", ");
		} 
		Console.print(" =?= ");
		for(int i=0; i<expected.length; i++) 
		{
			Console.print(expected[i]+", ");
		}
		boolean isEqual = true;
		for(int i=0; i<a.length && i<expected.length; i++) 
		{
			if(a[i]!=expected[i]) {isEqual=false;}
		}
		if(a.length!=expected.length) 
		{
			isEqual = false;
		}
		
		if(isEqual) 
		{
			Console.println("\n"+bar+"  Result "+a+" == expected "+expected+" -> test successful.");
		}
		else 
		{
			Console.println("\n"+bar+"  Result is: "+a+" -> test failed!");
			success = false;
		}
		Console.println(bar+"====================================================\n");
		if(success==false) 
		{
			return 0;
		}
		return 1;
	}
	
	public int testGraph_RelationalDeriviation(NVertex[] Structure, NVertex head, double[] expectedInput, double[] expectedRelational) 
	{
		for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().loadState();}
		for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().forward();}
		for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().loadState();  }
		for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().preBackward(BigInteger.ZERO);}
		for(int i=0; i<Structure.length; i++) {Structure[i].asExecutable().Backward(BigInteger.ZERO);}
		
		boolean success = true;
		Console.println("");
		Console.println("Root Structure Deriviation Test: ");
		Console.println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
		for(int i=0; i<Structure.length; i++) 
		{
			String expression = "null"; 
			if(Structure[i].getFunction()!=null) 
			{
				expression = Structure[i].getFunction().expression();
			}
			Console.print(bar+"  Root ["+i+"]: ");
			if(Structure[i]==head) {Console.println(expression+"  => (head)");}
			else 
			{
				Console.println(expression);
			}
		}
		Console.println(bar+"====================================================");
		//-----------------------------------------------------------------------
		Console.println(bar+"");
		Console.println(bar+"  Input Derivatives: ");
		Console.print(bar+" ");
		
		double[] a = head.getInputDerivative(0);
		for(int i=0; i<a.length; i++) 
		{
			Console.print(a[i]+", ");
		} 
		Console.print(" =?= ");
		for(int i=0; i<expectedInput.length; i++) 
		{
			Console.print(expectedInput[i]+", ");
		}
		boolean isEqual = true;
		for(int i=0; i<a.length && i<expectedInput.length; i++) 
		{
			if(a[i]!=expectedInput[i]) {isEqual=false;}
		}
		if(a.length!=expectedInput.length) 
		{
			isEqual = false;
		}
		
		if(isEqual) 
		{
			Console.println("\n"+bar+"  Result "+a+" == expected "+expectedInput+" -> test successful.");
		}
		else 
		{
			Console.println("\n"+bar+"  Result is: "+a+" -> test failed!");
			success = false;
		}
		//-----------------------------------------------------------------------
		Console.println(bar+"====================================================\n");
		Console.println(bar+"");
		Console.println(bar+"  Relational Derivatives: ");
		Console.print(bar+" ");
		
		a = head.getRelationalDerivative(0);
		for(int i=0; i<a.length; i++) 
		{
			Console.print(a[i]+", ");
		} 
		Console.print(" =?= ");
		for(int i=0; i<expectedRelational.length; i++) 
		{
			Console.print(expectedRelational[i]+", ");
		}
		isEqual = true;
		for(int i=0; i<a.length && i<expectedRelational.length; i++) 
		{
			if(a[i]!=expectedRelational[i]) {isEqual=false;}
		}
		if(a.length!=expectedRelational.length) 
		{
			isEqual = false;
		}
		
		if(isEqual) 
		{
			Console.println("\n"+bar+"  Result "+a+" == expected "+expectedRelational+" -> test successful.");
		}
		else 
		{
			Console.println("\n"+bar+"  Result is: "+a+" -> test failed!");
			success = false;
		}
		//-----------------------------------------------------------------------
		Console.println(bar+"====================================================\n");
		if(success==false) 
		{
			return 0;
		}
		return 1;
	}
	
	
}
